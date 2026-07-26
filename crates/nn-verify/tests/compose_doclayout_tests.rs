// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for the DocLayout-YOLO detection pipeline.
//!
//! Verifies IBP and CROWN bound propagation through DocLayout-YOLO sub-components
//! (YOLOv10-based document layout detection):
//!
//! ## Tests (10 tests)
//!
//! 1.  **Conv2d + BatchNorm bound composition** — ConvBnSiLU building block
//!     with frozen BN statistics (IBP + CROWN)
//! 2.  **Detection head output bounds** — Sigmoid classification head
//!     guarantees output in [0, 1] (IBP)
//! 3.  **Feature pyramid network bound propagation** — Multi-scale feature
//!     fusion through lateral convs + channel concat (IBP)
//! 4.  **NMS output range** — Score filtering with threshold produces
//!     non-negative outputs via ReLU (IBP)
//! 5.  **Backbone stride-2 downsampling** — Conv2d with stride-2 spatial
//!     reduction and channel expansion (IBP + CROWN)
//! 6.  **SPPF multi-scale pooling** — Cascaded MaxPool2d with channel
//!     concatenation preserves bounds (IBP)
//! 7.  **DFL box regression** — Distribution Focal Loss decoding via
//!     softmax + weighted sum (IBP)
//! 8.  **C2f residual block** — Entry conv + bottleneck with skip connection
//!     + concat + exit conv (IBP)
//! 9.  **End-to-end backbone + detection head** — Full pipeline from image
//!     to sigmoid-bounded confidence scores (IBP)
//! 10. **Monotone tightening** — Narrower pixel bounds produce no-wider
//!     output bounds (IBP soundness property)
//!
//! Architecture: DocLayout-YOLO (Zhao et al. 2024) based on YOLOv10.
//! - ConvBnAct: Conv2d -> BatchNorm -> SiLU (backbone building block)
//! - C2f: Multi-branch block with bottleneck residuals
//! - SPPF: Spatial Pyramid Pooling - Fast (cascaded MaxPool)
//! - DFL: Distribution Focal Loss for box regression
//! - PAN neck: Path Aggregation Network for multi-scale features
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_C=3, SPATIAL=8, CHANNELS=16, NUM_CLASSES=4, NUM_ANCHORS=64
//!
//! Part of #4186: Add compose tests for DocLayout-YOLO detection pipeline bounds.

mod common;

use common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Feature map spatial size.
const SPATIAL: usize = 8;
/// Primary channel width.
const CHANNELS: usize = 16;
/// Doubled channel width.
const CHANNELS_2X: usize = CHANNELS * 2;
/// Number of detection classes.
const NUM_CLASSES: usize = 4;
/// DFL regression bins.
const DFL_BINS: usize = 8;
/// Number of anchors (spatial^2 for single-scale).
const NUM_ANCHORS: usize = SPATIAL * SPATIAL;
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
    x: TensorNodeId,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    out_h: usize,
    out_w: usize,
) -> TensorNodeId {
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
// 1. Conv2d + BatchNorm bound composition (IBP + CROWN)
// ===========================================================================

/// Build ConvBnSiLU block: Conv2d -> BatchNorm -> SiLU.
///
/// Input: `[IN_CH, IMG_SIZE, IMG_SIZE]` (Variable, image [0, 1]).
/// Output: `[CHANNELS, IMG_SIZE, IMG_SIZE]` (feature map).
fn build_conv_bn_silu_kernel() -> TensorKernelDef {
    let s = IMG_SIZE;
    let mut b = TensorBlockBuilder::new("doclayout_test_conv_bn_silu");

    let input = b.add_input("image", &[IN_CH, s, s]);
    let out = add_conv_bn_silu(&mut b, input, "stem", IN_CH, CHANNELS, 3, 1, 1, s, s);

    b.build(out).expect("valid ConvBnSiLU kernel")
}

fn conv_bn_silu_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, IN_CH, 3);
    bindings
}

/// Verifies Conv2d + BatchNorm + SiLU produces finite bounded outputs.
#[test]
fn test_doclayout_conv_bn_silu_bounds_compose() {
    let def = build_conv_bn_silu_kernel();
    def.validate().expect("ConvBnSiLU should validate");

    let bindings = conv_bn_silu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP through ConvBnSiLU");
    assert_eq!(
        ibp_out.lower_upper().0.shape(),
        &[CHANNELS, IMG_SIZE, IMG_SIZE]
    );
    assert_bounds_valid(&ibp_out);
    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("DocLayout ConvBnSiLU IBP (image [0,1]): [{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());

    // CROWN
    let (method, crown_out, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("DocLayout ConvBnSiLU CROWN ({method:?}): [{clo}, {chi}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 2. Detection head output bounds (IBP)
// ===========================================================================

/// Build detection head: Conv2d -> reshape -> sigmoid.
///
/// Input: `[CHANNELS, SPATIAL, SPATIAL]` (Variable, backbone features).
/// Output: `[NUM_ANCHORS, NUM_CLASSES]` (class probabilities in [0, 1]).
///
/// Sigmoid guarantees output in (0, 1) for any finite input.
fn build_detection_head_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s = SPATIAL;
    let cls_shape = [NUM_CLASSES, s, s];
    let flat_shape = [NUM_ANCHORS, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("doclayout_test_detect_head");

    let input = b.add_input("features", &[c, s, s]);
    let cls_w = b.add_input("cls_conv_w", &[NUM_CLASSES, c, 1, 1]);
    let cls_b = b.add_input("cls_conv_b", &[NUM_CLASSES]);

    let cls_conv = b.add_conv2d(input, cls_w, Some(cls_b), 1, 1, 0, 0, &cls_shape);
    let reshaped = b.add_reshape(cls_conv, &flat_shape);
    let out = b.add_sigmoid(reshaped, &flat_shape);

    b.build(out).expect("valid detection head kernel")
}

fn detection_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, CHANNELS, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])),
    ]
}

/// Detection head sigmoid output must be bounded in [0, 1].
#[test]
fn test_doclayout_detection_head_bounds_compose() {
    let def = build_detection_head_kernel();
    def.validate().expect("detection head should validate");

    let bindings = detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detection head");
    assert_eq!(output.lower_upper().0.shape(), &[NUM_ANCHORS, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("DocLayout detection head IBP: [{lo}, {hi}]");

    // Sigmoid output must be in [0, 1].
    let eps = 1e-6;
    assert!(
        lo >= 0.0 - eps,
        "detection head sigmoid lower must be >= 0, got {lo}"
    );
    assert!(
        hi <= 1.0 + eps,
        "detection head sigmoid upper must be <= 1, got {hi}"
    );
}

// ===========================================================================
// 3. Feature pyramid network bound propagation (IBP)
// ===========================================================================

/// Build neck FPN top-down path.
///
/// Input 1 (Variable): `[CHANNELS, SPATIAL, SPATIAL]` (hi-res features).
/// Input 2 (Variable): `[CHANNELS_2X, SPATIAL/2, SPATIAL/2]` (lo-res features).
/// Output: `[CHANNELS, SPATIAL, SPATIAL]` (fused features).
///
/// Architecture:
///   lo -> Conv2d(2C, C, 1x1) -> BN -> SiLU -> reshape to [UP_C, S, S]
///   concat(hi, upsampled_lo) -> [C + UP_C, S, S]
///   Conv2d(C + UP_C, C, 1x1) -> BN -> SiLU -> [C, S, S]
///
/// A reshape preserves element count, so modeling the nearest-neighbor 2x
/// upsample of the `[C, S/2, S/2]` lateral output as a reshape to `[C, S, S]`
/// is invalid (4x more elements). We instead model it soundly as the
/// element-count-preserving reshape `[C, S/2, S/2] -> [C/4, S, S]` (the
/// established trick used in the deep PAN kernels), trading channels for
/// spatial resolution; the downstream concat/reduce channel counts follow.
fn build_fpn_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let c2 = CHANNELS_2X;
    let s = SPATIAL;
    let s_lo = s / 2;
    // Reshape preserves element count: c*s_lo*s_lo == up_c*s*s, so up_c = c/4.
    let up_c = c * s_lo * s_lo / (s * s);
    let concat_c = c + up_c;
    let hi_shape = [c, s, s];
    let lo_shape = [c2, s_lo, s_lo];
    let up_shape = [up_c, s, s];
    let concat_shape = [concat_c, s, s];
    let mut b = TensorBlockBuilder::new("doclayout_test_fpn");

    let hi_feat = b.add_input("hi_features", &hi_shape);
    let lo_feat = b.add_input("lo_features", &lo_shape);

    // Lateral 1x1 conv on lo-res features
    let lateral = add_conv_bn_silu(&mut b, lo_feat, "lateral", c2, c, 1, 1, 0, s_lo, s_lo);

    // Reshape to model nearest-neighbor upsample 2x (element-count preserving)
    let upsampled = b.add_reshape(lateral, &up_shape);

    // Concat along channel dim
    let concat = b.add_concat(&[hi_feat, upsampled], 0, &concat_shape);

    // Reduction conv
    let out = add_conv_bn_silu(&mut b, concat, "reduce", concat_c, c, 1, 1, 0, s, s);

    b.build(out).expect("valid FPN kernel")
}

fn fpn_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let c2 = CHANNELS_2X;
    let s = SPATIAL;
    let s_lo = s / 2;
    // Mirror the element-count-preserving upsample reshape in build_fpn_kernel:
    // upsampled channels = c/4, so the reduce conv consumes c + c/4 channels.
    let up_c = c * s_lo * s_lo / (s * s);
    let concat_c = c + up_c;
    let mut bindings = vec![
        TensorParamBinding::Variable, // hi_features
        TensorParamBinding::Variable, // lo_features
    ];
    push_conv_bn_silu_bindings(&mut bindings, c, c2, 1); // lateral
    push_conv_bn_silu_bindings(&mut bindings, c, concat_c, 1); // reduce
    bindings
}

/// FPN produces finite bounded fused features from multi-scale inputs.
#[test]
fn test_doclayout_fpn_bounds_compose() {
    let def = build_fpn_kernel();
    def.validate().expect("FPN should validate");

    let bindings = fpn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let s = SPATIAL;
    let s_lo = s / 2;
    let hi_flat = CHANNELS * s * s;
    let lo_flat = CHANNELS_2X * s_lo * s_lo;
    let input = uniform_bounds(&[hi_flat + lo_flat], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through FPN");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL, SPATIAL]
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("DocLayout FPN IBP: [{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 4. NMS output range (IBP)
// ===========================================================================

/// Build NMS score filtering: sigmoid -> subtract threshold -> ReLU.
///
/// Input: `[NUM_ANCHORS, NUM_CLASSES]` (Variable, classification logits).
/// Output: `[NUM_ANCHORS, NUM_CLASSES]` (filtered scores, >= 0).
///
/// ReLU guarantees non-negative output. Scores below threshold are zeroed.
fn build_nms_filter_kernel() -> TensorKernelDef {
    let shape = [NUM_ANCHORS, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("doclayout_test_nms_filter");

    let input = b.add_input("cls_logits", &shape);
    let conf = b.add_sigmoid(input, &shape);
    let thresh = b.add_input("threshold", &shape);
    let diff = b.add_binary_add(conf, thresh, &shape);
    let out = b.add_relu(diff, &shape);

    b.build(out).expect("valid NMS filter kernel")
}

fn nms_filter_bindings() -> Vec<TensorParamBinding> {
    let conf_threshold = 0.25f32;
    let thresh_data = ArrayD::from_elem(IxDyn(&[NUM_ANCHORS, NUM_CLASSES]), -conf_threshold);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(thresh_data),
    ]
}

/// NMS score filtering produces non-negative output via ReLU.
#[test]
fn test_doclayout_nms_output_range_compose() {
    let def = build_nms_filter_kernel();
    def.validate().expect("NMS filter should validate");

    let bindings = nms_filter_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, NUM_CLASSES], 5.0);

    let output = graph.propagate_ibp(&input).expect("IBP through NMS filter");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("DocLayout NMS filter IBP: [{lo}, {hi}]");

    // ReLU guarantees non-negative output.
    assert!(
        lo >= -1e-5,
        "NMS filtered scores must be >= 0 (ReLU), got {lo}"
    );
    // Upper bound: sigmoid max is 1, minus threshold 0.25 = 0.75.
    assert!(
        hi <= 1.0 - 0.25 + 1e-3,
        "NMS filtered upper should be <= 0.75, got {hi}"
    );
}

// ===========================================================================
// 5. Backbone stride-2 downsampling (IBP + CROWN)
// ===========================================================================

/// Build backbone stage with stride-2 spatial reduction.
///
/// Input: `[IN_CH, IMG_SIZE, IMG_SIZE]` (Variable, image [0, 1]).
/// Output: `[CHANNELS, IMG_SIZE/2, IMG_SIZE/2]`.
fn build_backbone_stage_kernel() -> TensorKernelDef {
    let s_out = IMG_SIZE / 2;
    let mut b = TensorBlockBuilder::new("doclayout_test_backbone_stage");

    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);
    let out = add_conv_bn_silu(&mut b, input, "s0", IN_CH, CHANNELS, 3, 2, 1, s_out, s_out);

    b.build(out).expect("valid backbone stage kernel")
}

fn backbone_stage_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, IN_CH, 3);
    bindings
}

/// Backbone stride-2 downsampling with IBP + CROWN.
#[test]
fn test_doclayout_backbone_stride2_bounds_compose() {
    let def = build_backbone_stage_kernel();
    def.validate().expect("backbone stage should validate");

    let bindings = backbone_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let s_out = IMG_SIZE / 2;

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP through backbone");
    assert_eq!(ibp_out.lower_upper().0.shape(), &[CHANNELS, s_out, s_out]);
    assert_bounds_valid(&ibp_out);
    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("DocLayout backbone stride-2 IBP: [{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());

    // CROWN
    let (method, crown_out, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("DocLayout backbone stride-2 CROWN ({method:?}): [{clo}, {chi}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 6. SPPF multi-scale pooling (IBP)
// ===========================================================================

/// Build SPPF block: cascaded MaxPool2d x3 + channel concat.
///
/// Input: `[CHANNELS, SPATIAL, SPATIAL]` (Variable, backbone features).
/// Output: `[CHANNELS * 4, SPATIAL, SPATIAL]` (multi-scale features).
///
/// SPPF concatenates: [input, pool1(input), pool2(pool1), pool3(pool2)]
/// along the channel dimension. Each MaxPool preserves spatial size.
fn build_sppf_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s = SPATIAL;
    let feat = [c, s, s];
    let sppf_shape = [c * 4, s, s];
    let mut b = TensorBlockBuilder::new("doclayout_test_sppf");

    let input = b.add_input("features", &feat);

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

    let out = b.add_concat(&[input, pool1, pool2, pool3], 0, &sppf_shape);

    b.build(out).expect("valid SPPF kernel")
}

fn sppf_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

/// SPPF preserves input bounds through max pooling and channel concat.
#[test]
fn test_doclayout_sppf_bounds_compose() {
    let def = build_sppf_kernel();
    def.validate().expect("SPPF should validate");

    let bindings = sppf_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through SPPF");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS * 4, SPATIAL, SPATIAL]
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("DocLayout SPPF IBP: [{lo}, {hi}]");

    // MaxPool preserves bounds: output range should be <= input range.
    // With input in [-2, 2], max pooled values stay in [-2, 2].
    assert!(
        lo >= -2.0 - 1e-5,
        "SPPF lower should be >= input lower, got {lo}"
    );
    assert!(
        hi <= 2.0 + 1e-5,
        "SPPF upper should be <= input upper, got {hi}"
    );
}

// ===========================================================================
// 7. DFL box regression (IBP)
// ===========================================================================

/// Build DFL (Distribution Focal Loss) box regression.
///
/// Input: `[NUM_ANCHORS, DFL_BINS]` (Variable, box regression logits).
/// Output: `[NUM_ANCHORS, 1]` (decoded box coordinate).
///
/// DFL: softmax over bins -> weighted sum with bin indices.
/// The softmax output is a probability distribution in [0, 1], and
/// the weighted sum produces a value in [0, DFL_BINS-1].
fn build_dfl_kernel() -> TensorKernelDef {
    let flat = [NUM_ANCHORS, DFL_BINS];
    let out_shape = [NUM_ANCHORS, 1];
    let mut b = TensorBlockBuilder::new("doclayout_test_dfl");

    let input = b.add_input("box_logits", &flat);
    let bins_w = b.add_input("dfl_bins", &[DFL_BINS, 1]);

    let probs = b.add_softmax(input, 1, &flat);
    let out = b.add_matmul(probs, bins_w, false, None, &out_shape);

    b.build(out).expect("valid DFL kernel")
}

fn dfl_bindings() -> Vec<TensorParamBinding> {
    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins"),
        ),
    ]
}

/// DFL box regression output is bounded by bin range [0, DFL_BINS-1].
#[test]
fn test_doclayout_dfl_box_regression_compose() {
    let def = build_dfl_kernel();
    def.validate().expect("DFL should validate");

    let bindings = dfl_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, DFL_BINS], 5.0);

    let output = graph.propagate_ibp(&input).expect("IBP through DFL");
    assert_eq!(output.lower_upper().0.shape(), &[NUM_ANCHORS, 1]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("DocLayout DFL IBP: [{lo}, {hi}]");

    // DFL output = weighted sum of bin indices [0, DFL_BINS-1] by softmax probs.
    // Minimum: all weight on bin 0 -> output = 0.
    // Maximum: all weight on bin (DFL_BINS-1) -> output = DFL_BINS-1.
    let max_bin = (DFL_BINS - 1) as f32;
    let eps = 1e-3;
    assert!(lo >= 0.0 - eps, "DFL lower bound should be >= 0, got {lo}");
    assert!(
        hi <= max_bin + eps,
        "DFL upper bound should be <= {max_bin}, got {hi}"
    );
}

// ===========================================================================
// 8. C2f residual block (IBP)
// ===========================================================================

/// Build C2f block: entry 1x1 conv -> bottleneck (3x3 + 3x3 + skip) ->
/// concat -> exit 1x1 conv.
///
/// Input: `[CHANNELS, SPATIAL, SPATIAL]` (Variable).
/// Output: `[CHANNELS, SPATIAL, SPATIAL]`.
fn build_c2f_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s = SPATIAL;
    let feat = [c, s, s];
    let mut b = TensorBlockBuilder::new("doclayout_test_c2f");

    let input = b.add_input("features", &feat);

    // Entry 1x1 conv
    let entry = add_conv_bn_silu(&mut b, input, "entry", c, c, 1, 1, 0, s, s);

    // Bottleneck: two 3x3 convs + skip
    let bn1 = add_conv_bn_silu(&mut b, entry, "bn1", c, c, 3, 1, 1, s, s);
    let bn2 = add_conv_bn_silu(&mut b, bn1, "bn2", c, c, 3, 1, 1, s, s);
    let residual = b.add_binary_add(bn2, entry, &feat);

    // Concat entry + bottleneck along channel dim
    let concat_shape = [c * 2, s, s];
    let concat_out = b.add_concat(&[entry, residual], 0, &concat_shape);

    // Exit 1x1 conv: reduce channels back to C
    let out = add_conv_bn_silu(&mut b, concat_out, "exit", c * 2, c, 1, 1, 0, s, s);

    b.build(out).expect("valid C2f kernel")
}

fn c2f_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, c, c, 1); // entry
    push_conv_bn_silu_bindings(&mut bindings, c, c, 3); // bn1
    push_conv_bn_silu_bindings(&mut bindings, c, c, 3); // bn2
    push_conv_bn_silu_bindings(&mut bindings, c, c * 2, 1); // exit
    bindings
}

/// C2f block with residual connection preserves finite bounds.
#[test]
fn test_doclayout_c2f_residual_bounds_compose() {
    let def = build_c2f_kernel();
    def.validate().expect("C2f should validate");

    let bindings = c2f_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through C2f");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL, SPATIAL]
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("DocLayout C2f IBP: [{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 9. End-to-end backbone + detection head (IBP)
// ===========================================================================

/// Build backbone -> SPPF -> detection head pipeline.
///
/// Input: `[IN_CH, IMG_SIZE, IMG_SIZE]` (Variable, image [0, 1]).
/// Output: `[NUM_ANCHORS, NUM_CLASSES]` (sigmoid-bounded confidence scores).
fn build_e2e_pipeline_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s = IMG_SIZE / 2; // after stride-2
    let feat = [c, s, s];
    let sppf_c = c * 4;
    let sppf_shape = [sppf_c, s, s];
    let cls_conv_shape = [NUM_CLASSES, s, s];
    let cls_flat = [NUM_ANCHORS, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("doclayout_test_e2e_pipeline");

    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);

    // Backbone: ConvBnSiLU stride-2
    let backbone = add_conv_bn_silu(&mut b, input, "backbone", IN_CH, c, 3, 2, 1, s, s);

    // SPPF
    let pool1 = b.add_max_pool_2d(
        backbone,
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
    let sppf = b.add_concat(&[backbone, pool1, pool2, pool3], 0, &sppf_shape);

    // Detection head: 1x1 conv -> reshape -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, sppf_c, 1, 1]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_conv = b.add_conv2d(sppf, cls_w, Some(cls_b), 1, 1, 0, 0, &cls_conv_shape);
    let cls_reshaped = b.add_reshape(cls_conv, &cls_flat);
    let out = b.add_sigmoid(cls_reshaped, &cls_flat);

    b.build(out).expect("valid e2e pipeline kernel")
}

fn e2e_pipeline_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let sppf_c = c * 4;
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, c, IN_CH, 3); // backbone
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        NUM_CLASSES,
        sppf_c,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])));
    bindings
}

/// End-to-end pipeline: image -> backbone -> SPPF -> sigmoid in [0, 1].
#[test]
fn test_doclayout_e2e_pipeline_bounds_compose() {
    let def = build_e2e_pipeline_kernel();
    def.validate().expect("e2e pipeline should validate");

    let bindings = e2e_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through e2e pipeline");
    assert_eq!(output.lower_upper().0.shape(), &[NUM_ANCHORS, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("DocLayout e2e pipeline IBP: [{lo}, {hi}]");

    // Sigmoid output must be in [0, 1].
    let eps = 1e-5;
    assert!(lo >= 0.0 - eps, "e2e sigmoid lower must be >= 0, got {lo}");
    assert!(hi <= 1.0 + eps, "e2e sigmoid upper must be <= 1, got {hi}");
}

// ===========================================================================
// 10. Monotone tightening (IBP)
// ===========================================================================

/// Verifies IBP monotonicity: narrower input pixel bounds produce output
/// bounds that are no wider than those from the full [0, 1] range.
///
/// This is a fundamental soundness property: if the input domain shrinks,
/// the output bounds cannot grow.
#[test]
fn test_doclayout_monotone_tightening_compose() {
    let def = build_e2e_pipeline_kernel();
    let bindings = e2e_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input: pixels in [0, 1].
    let wide_input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");

    // Narrow input: pixels in [0.2, 0.8].
    let narrow_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CH, IMG_SIZE, IMG_SIZE]), 0.2f32),
        ArrayD::from_elem(IxDyn(&[IN_CH, IMG_SIZE, IMG_SIZE]), 0.8f32),
    )
    .expect("valid narrow bounds");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");

    let (lo_w, hi_w) = bounds_min_max(&wide_output);
    let (lo_n, hi_n) = bounds_min_max(&narrow_output);
    let wide_width = hi_w - lo_w;
    let narrow_width = hi_n - lo_n;

    eprintln!(
        "DocLayout monotone tightening: wide=[{lo_w:.4}, {hi_w:.4}] w={wide_width:.4} \
         | narrow=[{lo_n:.4}, {hi_n:.4}] w={narrow_width:.4}"
    );

    // Monotonicity: narrow input bounds -> output bounds no wider.
    assert!(
        narrow_width <= wide_width + 1e-4,
        "monotone tightening violated: narrow_width={narrow_width} > wide_width={wide_width}"
    );
}
