// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep NY compose tests for DocLayout-YOLO subgraphs.
//!
//! These tests verify bounds propagation through intermediate-depth compositions
//! of the DocLayout-YOLO pipeline (YOLOv10-based document layout detection). They
//! bridge the gap between existing sub-block tests (in `compose_dpdf_doclayout_yolo.rs`)
//! and full end-to-end verification by exercising compositions at increasing depth:
//!
//! 1. **C2f block** — Entry conv -> bottleneck with residual -> concat -> exit conv.
//!    The core multi-branch block of YOLOv8/DocLayout-YOLO. Tests channel splitting,
//!    bottleneck residual paths, and channel reduction (IBP + CROWN).
//!
//! 2. **SPPF + detection head** — MaxPool chain -> concat -> Conv -> sigmoid.
//!    Cross-stage composition from spatial pyramid pooling to classification
//!    output. Output must be bounded in [0, 1] (IBP).
//!
//! 3. **Backbone stage** — ConvBnAct with stride-2 downsampling, testing spatial
//!    reduction and channel expansion through BN + SiLU (IBP + CROWN).
//!
//! 4. **Neck FPN** — Multi-scale feature concat with Conv channel reduction.
//!    Models the top-down PAN neck path: 1x1 conv + reshape + concat (IBP).
//!
//! 5. **Detection pipeline** — Backbone -> neck features -> dual heads
//!    (classification sigmoid + box DFL). Near-end-to-end composition
//!    verifying that cls output is in [0, 1] and box regression is finite (IBP).
//!
//! 6. **Widening analysis** — 1-stage vs 2-stage backbone bounds comparison.
//!    Quantifies bounds growth through depth to detect vacuous blowup (IBP).
//!
//! Architecture reference:
//! - DocLayout-YOLO (Zhao et al. 2024): Document layout detection based on YOLOv10
//! - YOLOv8 C2f block: Conv -> split -> N bottleneck residuals -> concat -> Conv
//! - SPPF: Spatial Pyramid Pooling - Fast from YOLOv5/YOLOv8
//! - DFL (Li et al. 2022): Distribution Focal Loss for box regression
//! - PAN neck: Path Aggregation Network for multi-scale feature fusion
//!
//! Dimensions are small for fast verification (CHANNELS=16, SPATIAL=8).
//! All tests use IbpValidated soundness mode per nn engineering rules
//! (Sound refuses linearization for normalization layers).
//!
//! Part of #3870: deep NY compose tests for DocLayout-YOLO.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Feature map spatial size.
const SPATIAL: usize = 8;
/// Primary channel width.
const CHANNELS: usize = 16;
/// Doubled channel width (after concat or expansion).
const CHANNELS_2X: usize = CHANNELS * 2; // 32
/// Number of detection classes.
const NUM_CLASSES: usize = 4;
/// DFL regression bins.
const DFL_BINS: usize = 8;
/// Number of anchors (spatial^2 for single-scale).
const NUM_ANCHORS: usize = SPATIAL * SPATIAL; // 64
/// Input image spatial size.
const IMG_SIZE: usize = 16;
/// Input channels (RGB).
const IN_CH: usize = 3;
/// SPPF MaxPool kernel size.
const SPPF_POOL_K: usize = 5;
/// SPPF padding (preserve spatial with k=5).
const SPPF_POOL_PAD: usize = 2;
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

/// Push ConvBnSiLU bindings (7 params: conv_w, conv_b, bn_mean, bn_var, bn_w, bn_b, eps).
fn push_conv_bn_silu_bindings(
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

/// Add a ConvBnSiLU block to the builder (7 input nodes).
///
/// Returns the output node ID. SiLU is decomposed as sigmoid(x) * x.
fn add_conv_bn_silu(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::TensorNodeId,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    out_h: usize,
    out_w: usize,
) -> nn_dsl::TensorNodeId {
    let out_shape = [out_ch, out_h, out_w];

    let cw = b.add_input(
        &format!("{prefix}_conv_w"),
        &[out_ch, in_ch, kernel, kernel],
    );
    let cb = b.add_input(&format!("{prefix}_conv_b"), &[out_ch]);
    let bm = b.add_input(&format!("{prefix}_bn_mean"), &[out_ch]);
    let bv = b.add_input(&format!("{prefix}_bn_var"), &[out_ch]);
    let bw = b.add_input(&format!("{prefix}_bn_w"), &[out_ch]);
    let bb = b.add_input(&format!("{prefix}_bn_b"), &[out_ch]);
    let eps = b.add_input(&format!("{prefix}_eps"), &[1]);

    let conv = b.add_conv2d(
        x,
        cw,
        Some(cb),
        stride,
        stride,
        padding,
        padding,
        &out_shape,
    );
    let bn = b.add_batch_norm(conv, bm, bv, bw, bb, eps, &out_shape);
    let sig = b.add_sigmoid(bn, &out_shape);
    b.add_binary_mul(bn, sig, &out_shape)
}

// ===========================================================================
// 1. C2f block: Entry conv -> bottleneck (Conv->BN->SiLU x2 + skip) ->
//    concat -> exit conv
// ===========================================================================

/// Build a C2f block kernel (YOLOv8 core multi-branch block).
///
/// Input: `[CHANNELS, SPATIAL, SPATIAL]` (Variable).
/// Output: `[CHANNELS, SPATIAL, SPATIAL]`.
///
/// Architecture:
///   1. Entry ConvBnSiLU(C, C, 1x1) — channel mixing
///   2. Bottleneck: ConvBnSiLU(C, C, 3x3) -> ConvBnSiLU(C, C, 3x3) + skip
///   3. Concat(entry_out, bottleneck_out) along channels -> [2C, S, S]
///   4. Exit ConvBnSiLU(2C, C, 1x1) — channel reduction
fn build_c2f_deep_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s = SPATIAL;
    let feat = [c, s, s];
    let mut b = TensorBlockBuilder::new("doclayout_deep_c2f");

    let input = b.add_input("features", &feat);

    // Entry 1x1 conv
    let entry = add_conv_bn_silu(&mut b, input, "entry", c, c, 1, 1, 0, s, s);

    // Bottleneck path: two 3x3 convs + residual
    let bn1 = add_conv_bn_silu(&mut b, entry, "bn1", c, c, 3, 1, 1, s, s);
    let bn2 = add_conv_bn_silu(&mut b, bn1, "bn2", c, c, 3, 1, 1, s, s);
    let residual = b.add_binary_add(bn2, entry, &feat);

    // Concat entry + bottleneck residual along channel dim
    let concat_shape = [c * 2, s, s];
    let concat_out = b.add_concat(&[entry, residual], 0, &concat_shape);

    // Exit 1x1 conv: reduce channels back to C
    let out = add_conv_bn_silu(&mut b, concat_out, "exit", c * 2, c, 1, 1, 0, s, s);

    b.build(out).expect("valid C2f deep kernel")
}

fn c2f_deep_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let mut bindings = vec![TensorParamBinding::Variable];
    // Entry 1x1
    push_conv_bn_silu_bindings(&mut bindings, c, c, 1);
    // Bottleneck conv 1
    push_conv_bn_silu_bindings(&mut bindings, c, c, 3);
    // Bottleneck conv 2
    push_conv_bn_silu_bindings(&mut bindings, c, c, 3);
    // Exit 1x1
    push_conv_bn_silu_bindings(&mut bindings, c, c * 2, 1);
    bindings
}

#[test]
fn test_doclayout_deep_c2f_ibp() {
    let def = build_c2f_deep_kernel();
    let bindings = c2f_deep_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through C2f");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL, SPATIAL],
        "C2f output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO deep C2f IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_doclayout_deep_c2f_crown() {
    let def = build_c2f_deep_kernel();
    let bindings = c2f_deep_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO deep C2f CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_doclayout_deep_c2f_verify_and_record() {
    let def = build_c2f_deep_kernel();
    let bindings = c2f_deep_bindings();
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let result = verify_and_assert(&def, &bindings, &input, "doclayout_deep_c2f");
    assert_eq!(result.num_variables, 1, "single Variable input");
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL, SPATIAL]
    );
}

// ===========================================================================
// 2. SPPF + detection head: MaxPool chain -> concat -> Conv -> sigmoid
// ===========================================================================

/// Build SPPF followed by a 1x1 conv reduction and sigmoid classification head.
///
/// Input: `[CHANNELS, SPATIAL, SPATIAL]` (Variable, backbone features).
/// Output: `[NUM_ANCHORS, NUM_CLASSES]` (class probabilities in [0, 1]).
///
/// Architecture:
///   SPPF: MaxPool chain x3 -> concat -> [4*C, S, S]
///   Conv2d(4*C, NUM_CLASSES, 1x1) -> reshape -> sigmoid
fn build_sppf_detection_head_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s = SPATIAL;
    let feat = [c, s, s];
    let sppf_out_c = c * 4; // 64
    let sppf_shape = [sppf_out_c, s, s];
    let cls_shape = [NUM_CLASSES, s, s];
    let flat_shape = [NUM_ANCHORS, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("doclayout_deep_sppf_detect");

    let input = b.add_input("features", &feat);

    // SPPF: cascaded MaxPool2d with same-padding
    let pool1 = b.add_max_pool_2d(
        input,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &feat,
    );
    let pool2 = b.add_max_pool_2d(
        pool1,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &feat,
    );
    let pool3 = b.add_max_pool_2d(
        pool2,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &feat,
    );

    // Concat input + pool1 + pool2 + pool3 along channel dim
    let sppf_out = b.add_concat(&[input, pool1, pool2, pool3], 0, &sppf_shape);

    // 1x1 Conv to classification logits
    let cls_w = b.add_input("cls_conv_w", &[NUM_CLASSES, sppf_out_c, 1, 1]);
    let cls_b = b.add_input("cls_conv_b", &[NUM_CLASSES]);
    let cls_conv = b.add_conv2d(sppf_out, cls_w, Some(cls_b), 1, 1, 0, 0, &cls_shape);

    // Reshape [NUM_CLASSES, S, S] -> [S*S, NUM_CLASSES] = [NUM_ANCHORS, NUM_CLASSES]
    let reshaped = b.add_reshape(cls_conv, &flat_shape);

    // Sigmoid: class probabilities in [0, 1]
    let out = b.add_sigmoid(reshaped, &flat_shape);

    b.build(out).expect("valid SPPF + detection head kernel")
}

fn sppf_detection_head_bindings() -> Vec<TensorParamBinding> {
    let sppf_out_c = CHANNELS * 4;
    vec![
        TensorParamBinding::Variable, // features
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, sppf_out_c, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])),
    ]
}

#[test]
fn test_doclayout_deep_sppf_detect_ibp() {
    let def = build_sppf_detection_head_kernel();
    let bindings = sppf_detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through SPPF+detect");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_ANCHORS, NUM_CLASSES],
        "SPPF+detect output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO deep SPPF+detect IBP: [{lo_min}, {hi_max}]");

    // Sigmoid output must be in [0, 1]
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

#[test]
fn test_doclayout_deep_sppf_detect_verify_and_record() {
    let def = build_sppf_detection_head_kernel();
    let bindings = sppf_detection_head_bindings();
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let result = verify_and_assert(&def, &bindings, &input, "doclayout_deep_sppf_detect");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[NUM_ANCHORS, NUM_CLASSES]
    );
}

// ===========================================================================
// 3. Backbone stage: ConvBnAct with stride-2 downsampling
// ===========================================================================

/// Build a single backbone stage with stride-2 downsampling.
///
/// Input: `[IN_CH, IMG_SIZE, IMG_SIZE]` (Variable, image [0, 1]).
/// Output: `[CHANNELS, IMG_SIZE/2, IMG_SIZE/2]`.
///
/// Architecture: Conv2d(3, C, 3, s=2, p=1) -> BN -> SiLU
fn build_backbone_stage_kernel() -> TensorKernelDef {
    let s_out = IMG_SIZE / 2; // 8
    let mut b = TensorBlockBuilder::new("doclayout_deep_backbone_stage");

    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);
    let out = add_conv_bn_silu(&mut b, input, "s0", IN_CH, CHANNELS, 3, 2, 1, s_out, s_out);

    b.build(out).expect("valid backbone stage kernel")
}

fn backbone_stage_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, IN_CH, 3);
    bindings
}

#[test]
fn test_doclayout_deep_backbone_stage_ibp() {
    let def = build_backbone_stage_kernel();
    let bindings = backbone_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let s_out = IMG_SIZE / 2;
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, s_out, s_out],
        "backbone stage output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO deep backbone stage IBP (image [0,1]): [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_doclayout_deep_backbone_stage_crown() {
    let def = build_backbone_stage_kernel();
    let bindings = backbone_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO deep backbone stage CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 4. Neck FPN: Multi-scale feature concat + Conv reduction
// ===========================================================================

/// Build a neck FPN (Feature Pyramid Network) top-down path.
///
/// Input 1 (Variable): `[CHANNELS, SPATIAL, SPATIAL]` (hi-res backbone features)
/// Input 2 (Variable): `[CHANNELS_2X, SPATIAL/2, SPATIAL/2]` (lo-res backbone features)
/// Output: `[CHANNELS, SPATIAL, SPATIAL]` (fused features).
///
/// Architecture:
///   lo -> Conv2d(2C, C, 1x1) -> BN -> SiLU -> reshape to [UP_C, S, S]
///   concat(hi, upsampled_lo) -> [C + UP_C, S, S]
///   Conv2d(C + UP_C, C, 1x1) -> BN -> SiLU -> [C, S, S]
///
/// A reshape preserves element count, so the nearest-neighbor 2x upsample of
/// the `[C, S/2, S/2]` lateral output is modeled soundly as the
/// element-count-preserving reshape `[C, S/2, S/2] -> [C/4, S, S]` (trading
/// channels for spatial resolution), not as a `[C, S, S]` reshape (which would
/// fabricate 4x the elements). Downstream concat/reduce channel counts follow.
fn build_neck_fpn_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let c2 = CHANNELS_2X;
    let s = SPATIAL;
    let s_lo = s / 2; // 4
    // Reshape preserves element count: c*s_lo*s_lo == up_c*s*s, so up_c = c/4.
    let up_c = c * s_lo * s_lo / (s * s);
    let concat_c = c + up_c;
    let hi_shape = [c, s, s];
    let lo_shape = [c2, s_lo, s_lo];
    let up_shape = [up_c, s, s];
    let concat_shape = [concat_c, s, s];
    let mut b = TensorBlockBuilder::new("doclayout_deep_neck_fpn");

    let hi_feat = b.add_input("hi_features", &hi_shape);
    let lo_feat = b.add_input("lo_features", &lo_shape);

    // 1x1 conv on lo-res to match hi-res channel count
    let lateral = add_conv_bn_silu(&mut b, lo_feat, "lateral", c2, c, 1, 1, 0, s_lo, s_lo);

    // Reshape to model nearest-neighbor upsample 2x (element-count preserving)
    let upsampled = b.add_reshape(lateral, &up_shape);

    // Concat hi-res and upsampled lo-res along channel dim
    let concat = b.add_concat(&[hi_feat, upsampled], 0, &concat_shape);

    // Reduction conv: (C + UP_C) -> C
    let out = add_conv_bn_silu(&mut b, concat, "reduce", concat_c, c, 1, 1, 0, s, s);

    b.build(out).expect("valid neck FPN kernel")
}

fn neck_fpn_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let c2 = CHANNELS_2X;
    let s = SPATIAL;
    let s_lo = s / 2;
    // Mirror the element-count-preserving upsample reshape: upsampled channels
    // = c/4, so the reduce conv consumes c + c/4 channels.
    let up_c = c * s_lo * s_lo / (s * s);
    let concat_c = c + up_c;
    let mut bindings = vec![
        TensorParamBinding::Variable, // hi_features
        TensorParamBinding::Variable, // lo_features
    ];
    // Lateral 1x1 conv
    push_conv_bn_silu_bindings(&mut bindings, c, c2, 1);
    // Reduction 1x1 conv
    push_conv_bn_silu_bindings(&mut bindings, c, concat_c, 1);
    bindings
}

#[test]
fn test_doclayout_deep_neck_fpn_ibp() {
    let def = build_neck_fpn_kernel();
    let bindings = neck_fpn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Multi-variable: concatenated flat input for both feature maps
    let s = SPATIAL;
    let s_lo = s / 2;
    let hi_flat = CHANNELS * s * s;
    let lo_flat = CHANNELS_2X * s_lo * s_lo;
    let input = uniform_bounds(&[hi_flat + lo_flat], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through neck FPN");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL, SPATIAL],
        "neck FPN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO deep neck FPN IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_doclayout_deep_neck_fpn_verify_and_record() {
    let def = build_neck_fpn_kernel();
    let bindings = neck_fpn_bindings();
    let s = SPATIAL;
    let s_lo = s / 2;
    let hi_flat = CHANNELS * s * s;
    let lo_flat = CHANNELS_2X * s_lo * s_lo;
    let input = uniform_bounds(&[hi_flat + lo_flat], 2.0);

    let result = verify_and_assert(&def, &bindings, &input, "doclayout_deep_neck_fpn");
    assert_eq!(
        result.num_variables, 2,
        "two Variable inputs (hi + lo features)"
    );
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL, SPATIAL]
    );
}

// ===========================================================================
// 5. Detection pipeline: Backbone -> neck -> dual heads (cls + box)
// ===========================================================================

/// Build a detection pipeline: 2-stage backbone -> SPPF -> dual heads.
///
/// Input: `[IN_CH, IMG_SIZE, IMG_SIZE]` (Variable, image [0, 1]).
/// Output: `[NUM_ANCHORS, NUM_CLASSES + 1]` (cls sigmoid + DFL box coordinate).
///
/// Architecture:
///   Stage 0: ConvBnSiLU(3, C, 3, s=2, p=1)  -> [C, S, S]
///   Stage 1: ConvBnSiLU(C, C, 3, s=1, p=1)   -> [C, S, S]
///   SPPF: MaxPool x3 -> concat -> [4C, S, S]
///   Cls head: Conv2d(4C, NUM_CLASSES, 1x1) -> reshape -> sigmoid
///   Box head: Conv2d(4C, DFL_BINS, 1x1) -> reshape -> softmax -> matmul(bins)
///   Concat cls + box along last dim -> [NUM_ANCHORS, NUM_CLASSES + 1]
fn build_detection_pipeline_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s = SPATIAL;
    let s_half = IMG_SIZE / 2; // 8 = SPATIAL
    let feat = [c, s, s];
    let sppf_c = c * 4;
    let sppf_shape = [sppf_c, s, s];
    let cls_conv_shape = [NUM_CLASSES, s, s];
    let cls_flat = [NUM_ANCHORS, NUM_CLASSES];
    let box_conv_shape = [DFL_BINS, s, s];
    let box_flat = [NUM_ANCHORS, DFL_BINS];
    let box_out = [NUM_ANCHORS, 1];
    let final_shape = [NUM_ANCHORS, NUM_CLASSES + 1];
    let mut b = TensorBlockBuilder::new("doclayout_deep_detect_pipeline");

    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);

    // Stage 0: stride-2 downsample
    let s0 = add_conv_bn_silu(&mut b, input, "s0", IN_CH, c, 3, 2, 1, s_half, s_half);

    // Stage 1: same spatial, deepen features
    let s1 = add_conv_bn_silu(&mut b, s0, "s1", c, c, 3, 1, 1, s, s);

    // SPPF
    let pool1 = b.add_max_pool_2d(
        s1,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &feat,
    );
    let pool2 = b.add_max_pool_2d(
        pool1,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &feat,
    );
    let pool3 = b.add_max_pool_2d(
        pool2,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &feat,
    );
    let sppf = b.add_concat(&[s1, pool1, pool2, pool3], 0, &sppf_shape);

    // Classification head: 1x1 conv -> reshape -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, sppf_c, 1, 1]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_conv = b.add_conv2d(sppf, cls_w, Some(cls_b), 1, 1, 0, 0, &cls_conv_shape);
    let cls_reshaped = b.add_reshape(cls_conv, &cls_flat);
    let cls_sigmoid = b.add_sigmoid(cls_reshaped, &cls_flat);

    // Box regression head: 1x1 conv -> reshape -> softmax -> matmul(bins)
    let box_w = b.add_input("box_w", &[DFL_BINS, sppf_c, 1, 1]);
    let box_b = b.add_input("box_b", &[DFL_BINS]);
    let bins_w = b.add_input("dfl_bins", &[DFL_BINS, 1]);
    let box_conv = b.add_conv2d(sppf, box_w, Some(box_b), 1, 1, 0, 0, &box_conv_shape);
    let box_reshaped = b.add_reshape(box_conv, &box_flat);
    let box_softmax = b.add_softmax(box_reshaped, 1, &box_flat);
    let box_dfl = b.add_matmul(box_softmax, bins_w, false, None, &box_out);

    // Concat cls + box along last dim: [NUM_ANCHORS, NUM_CLASSES + 1]
    let out = b.add_concat(&[cls_sigmoid, box_dfl], 1, &final_shape);

    b.build(out).expect("valid detection pipeline kernel")
}

fn detection_pipeline_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let sppf_c = c * 4;
    let mut bindings = vec![TensorParamBinding::Variable]; // image

    // Stage 0, Stage 1
    push_conv_bn_silu_bindings(&mut bindings, c, IN_CH, 3);
    push_conv_bn_silu_bindings(&mut bindings, c, c, 3);

    // Cls head: conv_w, conv_b
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        NUM_CLASSES,
        sppf_c,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])));

    // Box head: conv_w, conv_b, dfl_bins
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        DFL_BINS, sppf_c, 1, 1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[DFL_BINS])));
    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();
    bindings.push(TensorParamBinding::ConstantTensor(
        ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins"),
    ));

    bindings
}

#[test]
fn test_doclayout_deep_detect_pipeline_ibp() {
    let def = build_detection_pipeline_kernel();
    let bindings = detection_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detect pipeline");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_ANCHORS, NUM_CLASSES + 1],
        "detect pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO deep detect pipeline IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // Classification channels (first NUM_CLASSES) must be in [0, 1] (sigmoid)
    let (lo_arr, hi_arr) = output.lower_upper();
    let eps = 1e-5;
    for anchor in 0..NUM_ANCHORS {
        for cls in 0..NUM_CLASSES {
            let l = lo_arr[[anchor, cls]];
            let h = hi_arr[[anchor, cls]];
            assert!(
                l >= 0.0 - eps && h <= 1.0 + eps,
                "cls[{anchor},{cls}] out of [0,1]: [{l}, {h}]"
            );
        }
    }
}

#[test]
fn test_doclayout_deep_detect_pipeline_verify_and_record() {
    let def = build_detection_pipeline_kernel();
    let bindings = detection_pipeline_bindings();
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(&def, &bindings, &input, "doclayout_deep_detect_pipeline");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[NUM_ANCHORS, NUM_CLASSES + 1]
    );
}

// ===========================================================================
// 6. Widening analysis: 1-stage vs 2-stage backbone bounds comparison
// ===========================================================================

/// Build a 2-stage backbone (two ConvBnSiLU with stride-2).
///
/// Input: `[IN_CH, IMG_SIZE, IMG_SIZE]` (Variable, image [0, 1]).
/// Output: `[CHANNELS_2X, SPATIAL/2, SPATIAL/2]`.
fn build_backbone_two_stage_kernel() -> TensorKernelDef {
    let s0 = IMG_SIZE / 2;
    let s1 = IMG_SIZE / 4;
    let mut b = TensorBlockBuilder::new("doclayout_deep_backbone_2stage");

    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);
    let stage0 = add_conv_bn_silu(&mut b, input, "s0", IN_CH, CHANNELS, 3, 2, 1, s0, s0);
    let out = add_conv_bn_silu(&mut b, stage0, "s1", CHANNELS, CHANNELS_2X, 3, 2, 1, s1, s1);

    b.build(out).expect("valid 2-stage backbone kernel")
}

fn backbone_two_stage_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, IN_CH, 3);
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS_2X, CHANNELS, 3);
    bindings
}

/// Widening analysis: compare 1-stage vs 2-stage IBP bounds width.
///
/// Adding depth (more ConvBnSiLU stages) should widen IBP bounds. This test
/// quantifies the growth factor to detect vacuous blowup. A ratio > 100x
/// suggests the pipeline would benefit from CROWN or tighter input ranges.
#[test]
fn test_doclayout_deep_widening_1_vs_2_stages() {
    // 1-stage backbone
    let def1 = build_backbone_stage_kernel();
    let bindings1 = backbone_stage_bindings();
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph 1-stage");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let out1 = graph1.propagate_ibp(&input).expect("IBP 1-stage");
    assert_bounds_valid(&out1);
    let (lo1, hi1) = bounds_min_max(&out1);
    let width1 = hi1 - lo1;

    // 2-stage backbone
    let def2 = build_backbone_two_stage_kernel();
    let bindings2 = backbone_two_stage_bindings();
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("graph 2-stage");

    let out2 = graph2.propagate_ibp(&input).expect("IBP 2-stage");
    assert_bounds_valid(&out2);
    let (lo2, hi2) = bounds_min_max(&out2);
    let width2 = hi2 - lo2;

    eprintln!("DocLayout-YOLO widening analysis:");
    eprintln!("  1-stage: [{lo1}, {hi1}] width={width1}");
    eprintln!("  2-stage: [{lo2}, {hi2}] width={width2}");

    // 2-stage should be wider than 1-stage (more depth = more uncertainty)
    assert!(
        width2 >= width1 - 1e-6,
        "2-stage bounds ({width2}) should be at least as wide as 1-stage ({width1})"
    );

    // Ensure neither is vacuously wide (> 1e6)
    assert!(width1 < 1e6, "1-stage bounds vacuously wide: {width1}");
    assert!(width2 < 1e6, "2-stage bounds vacuously wide: {width2}");

    if width1 > 0.0 {
        let ratio = width2 / width1;
        eprintln!("  Width ratio (2-stage/1-stage): {ratio:.2}x");
    }
}

/// 2-stage backbone IBP standalone test.
#[test]
fn test_doclayout_deep_backbone_2stage_ibp() {
    let def = build_backbone_two_stage_kernel();
    let bindings = backbone_two_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let s1 = IMG_SIZE / 4;
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS_2X, s1, s1],
        "2-stage backbone output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO deep 2-stage backbone IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

/// 2-stage backbone CROWN test.
#[test]
fn test_doclayout_deep_backbone_2stage_crown() {
    let def = build_backbone_two_stage_kernel();
    let bindings = backbone_two_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO deep 2-stage backbone CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}
