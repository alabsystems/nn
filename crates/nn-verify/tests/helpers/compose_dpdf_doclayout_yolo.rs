// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: DocLayout-YOLO object detection model NY composition.
//!
//! Verifies bounds propagation through DocLayout-YOLO sub-blocks used in the
//! dpdf document layout analysis pipeline:
//!
//! 1. **ConvBnAct**: Conv2d -> BatchNorm -> SiLU activation
//!    Core building block of the YOLOv10/DocLayout-YOLO backbone.
//!
//! 2. **SPPF (Spatial Pyramid Pooling - Fast)**: MaxPool2d chain with concat
//!    Used at the end of the backbone to aggregate multi-scale features.
//!
//! 3. **Detection sigmoid**: Sigmoid classification head
//!    Final activation for object class probabilities, must output [0, 1].
//!
//! 4. **DFL regression**: Softmax -> weighted sum (Distribution Focal Loss decode)
//!    Converts DFL logits to continuous bounding box coordinates.
//!
//! Architecture references:
//! - DocLayout-YOLO (Zhao et al. 2024): Document layout detection based on YOLOv10
//! - SPPF: Spatial Pyramid Pooling - Fast from YOLOv5/YOLOv8
//! - DFL (Li et al. 2022): Distribution Focal Loss for box regression
//!
//! Dimensions (small for fast verification):
//! - Feature maps: 32x32 input, 16x16 after stride-2 conv
//! - Channels: 3 (input) -> 16 (backbone) -> 256 (SPPF)
//!
//! Part of #3870: NY compose tests for dpdf models.

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

/// Input image spatial size.
const IMG_SIZE: usize = 32;
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;
/// First-stage output channels after ConvBnAct.
const CONV_OUT_CHANNELS: usize = 16;
/// Spatial size after stride-2 convolution.
const CONV_OUT_SIZE: usize = IMG_SIZE / 2; // 16
/// SPPF input/output channels.
const SPPF_CHANNELS: usize = 64;
/// SPPF spatial size.
const SPPF_SIZE: usize = 8;
/// SPPF MaxPool kernel size.
const SPPF_POOL_K: usize = 5;
/// SPPF padding (to preserve spatial size with k=5).
const SPPF_POOL_PAD: usize = 2;
/// Number of detection anchors (query positions).
const NUM_ANCHORS: usize = 16;
/// Number of classes for detection head.
const NUM_CLASSES: usize = 10;
/// DFL regression bins.
const DFL_BINS: usize = 16;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// 1. ConvBnAct: Conv2d -> BatchNorm -> SiLU
// ===========================================================================

/// Build a ConvBnAct kernel (DocLayout-YOLO backbone building block).
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image pixels in [0, 1]).
/// Output: `[CONV_OUT_CHANNELS, CONV_OUT_SIZE, CONV_OUT_SIZE]`.
///
/// Architecture: Conv2d(3, 16, k=3, s=2, p=1) -> BatchNorm -> SiLU
///
/// SiLU = x * sigmoid(x), decomposed as sigmoid + binary_mul since
/// TensorBlockBuilder has no native add_silu.
fn build_doclayout_conv_bn_act_kernel() -> TensorKernelDef {
    let c_out = CONV_OUT_CHANNELS;
    let s_out = CONV_OUT_SIZE;
    let out_shape = [c_out, s_out, s_out];
    let mut b = TensorBlockBuilder::new("doclayout_conv_bn_act");

    // Conv2d inputs
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("conv_weight", &[c_out, IN_CHANNELS, 3, 3]);
    let conv_b = b.add_input("conv_bias", &[c_out]);

    // BatchNorm inputs
    let bn_mean = b.add_input("bn_running_mean", &[c_out]);
    let bn_var = b.add_input("bn_running_var", &[c_out]);
    let bn_weight = b.add_input("bn_weight", &[c_out]);
    let bn_bias = b.add_input("bn_bias", &[c_out]);
    let bn_eps = b.add_input("bn_eps", &[1]);

    // Conv2d: [3, 32, 32] -> [16, 16, 16] (stride=2, padding=1)
    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        2, // stride_h
        2, // stride_w
        1, // padding_h
        1, // padding_w
        &out_shape,
    );

    // BatchNorm: [16, 16, 16] -> [16, 16, 16]
    let bn_out = b.add_batch_norm(
        conv_out, bn_mean, bn_var, bn_weight, bn_bias, bn_eps, &out_shape,
    );

    // SiLU(x) = x * sigmoid(x): decomposed into sigmoid + binary_mul
    let sig = b.add_sigmoid(bn_out, &out_shape);
    let out = b.add_binary_mul(bn_out, sig, &out_shape);

    b.build(out).expect("valid DocLayout-YOLO ConvBnAct kernel")
}

/// Create image-domain input bounds: pixels in [0, 1].
fn image_bounds_01(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Bindings for ConvBnAct.
fn doclayout_conv_bn_act_bindings() -> Vec<TensorParamBinding> {
    let c_out = CONV_OUT_CHANNELS;
    let conv_w = ArrayD::from_elem(IxDyn(&[c_out, IN_CHANNELS, 3, 3]), WEIGHT_MAG);
    let conv_b = ArrayD::from_elem(IxDyn(&[c_out]), 0.0f32);
    let bn_mean = ArrayD::from_elem(IxDyn(&[c_out]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[c_out]), 1.0f32);
    let bn_weight = ArrayD::from_elem(IxDyn(&[c_out]), 1.0f32);
    let bn_bias = ArrayD::from_elem(IxDyn(&[c_out]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                  // image [3, 32, 32]
        TensorParamBinding::ConstantTensor(conv_w),    // conv_weight [16, 3, 3, 3]
        TensorParamBinding::ConstantTensor(conv_b),    // conv_bias [16]
        TensorParamBinding::ConstantTensor(bn_mean),   // bn_running_mean [16]
        TensorParamBinding::ConstantTensor(bn_var),    // bn_running_var [16]
        TensorParamBinding::ConstantTensor(bn_weight), // bn_weight [16]
        TensorParamBinding::ConstantTensor(bn_bias),   // bn_bias [16]
        TensorParamBinding::ConstantScalar(1e-5),      // bn_eps [1]
    ]
}

/// ConvBnAct TensorKernelDef validates.
#[test]
fn test_doclayout_conv_bn_act_def_validates() {
    let def = build_doclayout_conv_bn_act_kernel();
    def.validate()
        .expect("DocLayout-YOLO ConvBnAct kernel should validate");
}

/// ConvBnAct translates to NY GraphNetwork.
#[test]
fn test_doclayout_conv_bn_act_graph_builds() {
    let def = build_doclayout_conv_bn_act_kernel();
    let bindings = doclayout_conv_bn_act_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("ConvBnAct graph should translate");

    // Conv2d + BatchNorm + Sigmoid + BinaryMul = at least 4 nodes
    assert!(
        graph.num_nodes() >= 4,
        "ConvBnAct graph should have >= 4 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through ConvBnAct with [0, 1] image input.
#[test]
fn test_doclayout_conv_bn_act_ibp_bounds() {
    let def = build_doclayout_conv_bn_act_kernel();
    let bindings = doclayout_conv_bn_act_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DocLayout-YOLO ConvBnAct");

    let out_shape = [CONV_OUT_CHANNELS, CONV_OUT_SIZE, CONV_OUT_SIZE];
    assert_eq!(
        output.lower_upper().0.shape(),
        &out_shape,
        "ConvBnAct output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO ConvBnAct IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds propagate through ConvBnAct.
///
/// Conv2d is linear, BatchNorm is affine (at inference with running stats),
/// and SiLU (sigmoid * x) requires CROWN linearization for the sigmoid term.
#[test]
fn test_doclayout_conv_bn_act_crown_propagation() {
    let def = build_doclayout_conv_bn_act_kernel();
    let bindings = doclayout_conv_bn_act_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let out_shape = [CONV_OUT_CHANNELS, CONV_OUT_SIZE, CONV_OUT_SIZE];
    assert_eq!(output.lower_upper().0.shape(), &out_shape,);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO ConvBnAct: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record ConvBnAct.
#[test]
fn test_doclayout_conv_bn_act_verify_and_record() {
    let def = build_doclayout_conv_bn_act_kernel();
    let bindings = doclayout_conv_bn_act_bindings();
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(&def, &bindings, &input, "doclayout_conv_bn_act");
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let out_shape = [CONV_OUT_CHANNELS, CONV_OUT_SIZE, CONV_OUT_SIZE];
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &out_shape);
}

// ===========================================================================
// 2. SPPF: MaxPool2d chain with concatenation
// ===========================================================================

/// Build an SPPF kernel (Spatial Pyramid Pooling - Fast).
///
/// Input: `[SPPF_CHANNELS, SPPF_SIZE, SPPF_SIZE]` (Variable, feature map).
/// Output: `[SPPF_CHANNELS * 4, SPPF_SIZE, SPPF_SIZE]` after concat.
///
/// SPPF architecture (YOLOv5):
///   pool1 = MaxPool2d(input, k=5, s=1, p=2)    -- preserves spatial
///   pool2 = MaxPool2d(pool1, k=5, s=1, p=2)
///   pool3 = MaxPool2d(pool2, k=5, s=1, p=2)
///   output = concat(input, pool1, pool2, pool3)  -- along channel dim
///
/// The cascaded max-pools with same-padding aggregate multi-scale context
/// without changing spatial dimensions, making them concatenatable.
fn build_doclayout_sppf_kernel() -> TensorKernelDef {
    let c = SPPF_CHANNELS;
    let s = SPPF_SIZE;
    let feat_shape = [c, s, s];
    let out_channels = c * 4;
    let out_shape = [out_channels, s, s];
    let mut b = TensorBlockBuilder::new("doclayout_sppf");

    let input = b.add_input("features", &feat_shape);

    // MaxPool2d chain: k=5, s=1, p=2 (preserves spatial dimensions)
    let pool1 = b.add_max_pool_2d(
        input,
        SPPF_POOL_K,   // kernel_h
        SPPF_POOL_K,   // kernel_w
        1,             // stride_h
        1,             // stride_w
        SPPF_POOL_PAD, // padding_h
        SPPF_POOL_PAD, // padding_w
        &feat_shape,
    );
    let pool2 = b.add_max_pool_2d(
        pool1,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &feat_shape,
    );
    let pool3 = b.add_max_pool_2d(
        pool2,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &feat_shape,
    );

    // Concatenate along channel dimension (dim=0 for [C, H, W])
    let out = b.add_concat(
        &[input, pool1, pool2, pool3],
        0, // concat along channel dim
        &out_shape,
    );

    b.build(out).expect("valid DocLayout-YOLO SPPF kernel")
}

/// Bindings for SPPF (no learnable parameters, only the variable input).
fn doclayout_sppf_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // features [SPPF_CHANNELS, SPPF_SIZE, SPPF_SIZE]
    ]
}

/// SPPF TensorKernelDef validates.
#[test]
fn test_doclayout_sppf_def_validates() {
    let def = build_doclayout_sppf_kernel();
    def.validate()
        .expect("DocLayout-YOLO SPPF kernel should validate");
}

/// SPPF translates to NY GraphNetwork.
#[test]
fn test_doclayout_sppf_graph_builds() {
    let def = build_doclayout_sppf_kernel();
    let bindings = doclayout_sppf_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("SPPF graph should translate");

    // 3 MaxPool2d + 1 Concat = at least 4 nodes
    assert!(
        graph.num_nodes() >= 4,
        "SPPF graph should have >= 4 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through SPPF with [-2, 2] feature map input.
///
/// MaxPool2d is monotonic: max(lower) <= output <= max(upper).
/// With uniform [-2, 2] input, MaxPool should produce bounds within [-2, 2]
/// for each pool stage. Concatenation preserves per-element bounds.
#[test]
fn test_doclayout_sppf_ibp_bounds() {
    let def = build_doclayout_sppf_kernel();
    let bindings = doclayout_sppf_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SPPF_CHANNELS, SPPF_SIZE, SPPF_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DocLayout-YOLO SPPF");

    let out_channels = SPPF_CHANNELS * 4;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[out_channels, SPPF_SIZE, SPPF_SIZE],
        "SPPF output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO SPPF IBP (features [-2,2]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // MaxPool does not expand bounds beyond input range.
    assert!(
        lo_min >= -2.0 - 1e-6,
        "SPPF lower should be >= -2.0 (input lower), got {lo_min}"
    );
    assert!(
        hi_max <= 2.0 + 1e-6,
        "SPPF upper should be <= 2.0 (input upper), got {hi_max}"
    );
}

/// CROWN bounds propagate through SPPF.
///
/// MaxPool2d is piecewise-linear (selects max element), so CROWN can
/// linearize it. The concatenation is structure-only (no computation).
#[test]
fn test_doclayout_sppf_crown_propagation() {
    let def = build_doclayout_sppf_kernel();
    let bindings = doclayout_sppf_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SPPF_CHANNELS, SPPF_SIZE, SPPF_SIZE], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let out_channels = SPPF_CHANNELS * 4;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[out_channels, SPPF_SIZE, SPPF_SIZE],
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO SPPF: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record SPPF.
#[test]
fn test_doclayout_sppf_verify_and_record() {
    let def = build_doclayout_sppf_kernel();
    let bindings = doclayout_sppf_bindings();
    let input = uniform_bounds(&[SPPF_CHANNELS, SPPF_SIZE, SPPF_SIZE], 2.0);

    let result = verify_and_assert(&def, &bindings, &input, "doclayout_sppf");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let out_channels = SPPF_CHANNELS * 4;
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[out_channels, SPPF_SIZE, SPPF_SIZE]);
}

// ===========================================================================
// 3. Detection sigmoid: Sigmoid classification head
// ===========================================================================

/// Build a detection sigmoid classification head.
///
/// Input: `[NUM_ANCHORS, NUM_CLASSES]` (Variable, raw logits in [-10, 10]).
/// Output: `[NUM_ANCHORS, NUM_CLASSES]` (class probabilities in [0, 1]).
///
/// The sigmoid activation is the final step for multi-label object detection
/// classification. Unlike softmax (which normalizes across classes), sigmoid
/// treats each class independently -- standard in YOLO-family detectors.
fn build_doclayout_detection_sigmoid_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("doclayout_detection_sigmoid");

    let input = b.add_input("raw_logits", &[NUM_ANCHORS, NUM_CLASSES]);
    let out = b.add_sigmoid(input, &[NUM_ANCHORS, NUM_CLASSES]);

    b.build(out)
        .expect("valid DocLayout-YOLO detection sigmoid kernel")
}

/// Bindings for detection sigmoid.
fn doclayout_detection_sigmoid_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // raw_logits [NUM_ANCHORS, NUM_CLASSES]
    ]
}

/// Detection sigmoid TensorKernelDef validates.
#[test]
fn test_doclayout_detection_sigmoid_def_validates() {
    let def = build_doclayout_detection_sigmoid_kernel();
    def.validate()
        .expect("detection sigmoid kernel should validate");
}

/// Detection sigmoid translates to NY GraphNetwork.
#[test]
fn test_doclayout_detection_sigmoid_graph_builds() {
    let def = build_doclayout_detection_sigmoid_kernel();
    let bindings = doclayout_detection_sigmoid_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("detection sigmoid graph should translate");

    assert!(
        graph.num_nodes() >= 1,
        "detection sigmoid graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through detection sigmoid.
///
/// Sigmoid maps R -> (0, 1). With input [-10, 10], IBP should produce
/// output bounds within [0, 1] (sigmoid(-10) ~ 0, sigmoid(10) ~ 1).
#[test]
fn test_doclayout_detection_sigmoid_ibp_bounds() {
    let def = build_doclayout_detection_sigmoid_kernel();
    let bindings = doclayout_detection_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, NUM_CLASSES], 10.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detection sigmoid");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_ANCHORS, NUM_CLASSES],
        "detection sigmoid output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "DocLayout-YOLO detection sigmoid IBP (logits [-10,10]): bounds=[{lo_min}, {hi_max}]"
    );

    // Sigmoid codomain is (0, 1). IBP must respect this.
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper bound must be <= 1, got {hi_max}"
    );
}

/// CROWN bounds propagate through detection sigmoid.
///
/// Sigmoid is smooth and CROWN-friendly. CROWN linearization should
/// produce tighter bounds than IBP for the sigmoid function.
#[test]
fn test_doclayout_detection_sigmoid_crown_propagation() {
    let def = build_doclayout_detection_sigmoid_kernel();
    let bindings = doclayout_detection_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, NUM_CLASSES], 10.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_ANCHORS, NUM_CLASSES],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO detection sigmoid: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Even under CROWN, sigmoid bounds must be in [0, 1].
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower bound must be >= 0 under CROWN, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper bound must be <= 1 under CROWN, got {hi_max}"
    );
}

/// Verify and record detection sigmoid.
#[test]
fn test_doclayout_detection_sigmoid_verify_and_record() {
    let def = build_doclayout_detection_sigmoid_kernel();
    let bindings = doclayout_detection_sigmoid_bindings();
    let input = uniform_bounds(&[NUM_ANCHORS, NUM_CLASSES], 10.0);

    let result = verify_and_assert(&def, &bindings, &input, "doclayout_detection_sigmoid");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_ANCHORS, NUM_CLASSES]);
}

// ===========================================================================
// 4. DFL regression: Softmax -> weighted sum (Distribution Focal Loss decode)
// ===========================================================================

/// Build a DFL regression kernel (Distribution Focal Loss box decoding).
///
/// Input: `[NUM_ANCHORS, DFL_BINS]` (Variable, DFL logits in [-5, 5]).
/// Output: `[NUM_ANCHORS, 1]` (continuous box coordinate).
///
/// DFL architecture (Li et al. 2022):
///   probs = softmax(logits, dim=-1)     [NUM_ANCHORS, DFL_BINS]
///   coord = matmul(probs, bins)         [NUM_ANCHORS, 1]
///
/// where bins = [0, 1, 2, ..., DFL_BINS-1] is a fixed integer sequence.
/// The softmax converts logits to a distribution over discrete bin positions,
/// and the weighted sum recovers a continuous coordinate.
///
/// Key verification property: output should be bounded by [0, DFL_BINS-1]
/// since softmax produces a valid probability distribution.
fn build_doclayout_dfl_regression_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("doclayout_dfl_regression");

    let input = b.add_input("dfl_logits", &[NUM_ANCHORS, DFL_BINS]);
    let bins = b.add_input("bins", &[DFL_BINS, 1]);

    // Softmax along last dimension (dim=1 for [NUM_ANCHORS, DFL_BINS])
    let probs = b.add_softmax(input, 1, &[NUM_ANCHORS, DFL_BINS]);

    // Weighted sum: matmul(probs, bins) = [NUM_ANCHORS, 1]
    let out = b.add_matmul(probs, bins, false, None, &[NUM_ANCHORS, 1]);

    b.build(out)
        .expect("valid DocLayout-YOLO DFL regression kernel")
}

/// Bindings for DFL regression.
///
/// The bins tensor is constant: [0, 1, 2, ..., DFL_BINS-1] as a column vector.
fn doclayout_dfl_regression_bindings() -> Vec<TensorParamBinding> {
    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();
    let bins = ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins shape");

    vec![
        TensorParamBinding::Variable, // dfl_logits [NUM_ANCHORS, DFL_BINS]
        TensorParamBinding::ConstantTensor(bins), // bins [DFL_BINS, 1]
    ]
}

/// DFL regression TensorKernelDef validates.
#[test]
fn test_doclayout_dfl_regression_def_validates() {
    let def = build_doclayout_dfl_regression_kernel();
    def.validate()
        .expect("DFL regression kernel should validate");
}

/// DFL regression translates to NY GraphNetwork.
#[test]
fn test_doclayout_dfl_regression_graph_builds() {
    let def = build_doclayout_dfl_regression_kernel();
    let bindings = doclayout_dfl_regression_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("DFL regression graph should translate");

    // Softmax + MatMul = at least 2 nodes
    assert!(
        graph.num_nodes() >= 2,
        "DFL regression graph should have >= 2 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through DFL regression.
///
/// Softmax produces a probability distribution (sums to 1, all >= 0).
/// Weighted sum with bins [0, ..., DFL_BINS-1] should produce output
/// in [0, DFL_BINS-1].
#[test]
fn test_doclayout_dfl_regression_ibp_bounds() {
    let def = build_doclayout_dfl_regression_kernel();
    let bindings = doclayout_dfl_regression_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, DFL_BINS], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DFL regression");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_ANCHORS, 1],
        "DFL regression output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO DFL regression IBP (logits [-5,5]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Softmax output is a probability distribution over bins [0, 15].
    // The weighted sum should ideally be in [0, 15], but IBP may be wider.
    // We verify finiteness and non-vacuity here; tighter bounds via CROWN.
}

/// CROWN bounds propagate through DFL regression.
///
/// Softmax is piecewise-smooth and CROWN can linearize it. The matmul
/// with constant bins is a linear operation. CROWN should produce
/// tighter bounds than IBP.
#[test]
fn test_doclayout_dfl_regression_crown_propagation() {
    let def = build_doclayout_dfl_regression_kernel();
    let bindings = doclayout_dfl_regression_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, DFL_BINS], 5.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_ANCHORS, 1],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO DFL regression: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record DFL regression.
#[test]
fn test_doclayout_dfl_regression_verify_and_record() {
    let def = build_doclayout_dfl_regression_kernel();
    let bindings = doclayout_dfl_regression_bindings();
    let input = uniform_bounds(&[NUM_ANCHORS, DFL_BINS], 5.0);

    let result = verify_and_assert(&def, &bindings, &input, "doclayout_dfl_regression");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_ANCHORS, 1]);
}

// ===========================================================================
// 5. Bottleneck residual: Conv -> Conv + skip connection
// ===========================================================================

/// Bottleneck channels used in C2f blocks.
const BOTTLENECK_CHANNELS: usize = 16;
/// Spatial size for bottleneck tests (after downsampling).
const BOTTLENECK_SIZE: usize = 8;

/// Build a Bottleneck residual block (C2f building block).
///
/// Input: `[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE]` (Variable).
/// Output: `[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE]`.
///
/// Architecture (YOLOv8 Bottleneck with shortcut=true):
///   Conv2d(C, C, 3, s=1, p=1) -> BN -> SiLU
///   Conv2d(C, C, 3, s=1, p=1) -> BN -> SiLU
///   + skip connection (input added to output)
fn build_bottleneck_residual_kernel() -> TensorKernelDef {
    let c = BOTTLENECK_CHANNELS;
    let s = BOTTLENECK_SIZE;
    let feat_shape = [c, s, s];
    let mut b = TensorBlockBuilder::new("doclayout_bottleneck_residual");

    let input = b.add_input("features", &feat_shape);

    // First ConvBnAct: Conv2d(C, C, 3, s=1, p=1) -> BN -> SiLU
    let conv1_w = b.add_input("conv1_weight", &[c, c, 3, 3]);
    let conv1_b = b.add_input("conv1_bias", &[c]);
    let bn1_mean = b.add_input("bn1_running_mean", &[c]);
    let bn1_var = b.add_input("bn1_running_var", &[c]);
    let bn1_weight = b.add_input("bn1_weight", &[c]);
    let bn1_bias = b.add_input("bn1_bias", &[c]);
    let bn1_eps = b.add_input("bn1_eps", &[1]);

    let conv1_out = b.add_conv2d(input, conv1_w, Some(conv1_b), 1, 1, 1, 1, &feat_shape);
    let bn1_out = b.add_batch_norm(
        conv1_out,
        bn1_mean,
        bn1_var,
        bn1_weight,
        bn1_bias,
        bn1_eps,
        &feat_shape,
    );
    let sig1 = b.add_sigmoid(bn1_out, &feat_shape);
    let silu1 = b.add_binary_mul(bn1_out, sig1, &feat_shape);

    // Second ConvBnAct: Conv2d(C, C, 3, s=1, p=1) -> BN -> SiLU
    let conv2_w = b.add_input("conv2_weight", &[c, c, 3, 3]);
    let conv2_b = b.add_input("conv2_bias", &[c]);
    let bn2_mean = b.add_input("bn2_running_mean", &[c]);
    let bn2_var = b.add_input("bn2_running_var", &[c]);
    let bn2_weight = b.add_input("bn2_weight", &[c]);
    let bn2_bias = b.add_input("bn2_bias", &[c]);
    let bn2_eps = b.add_input("bn2_eps", &[1]);

    let conv2_out = b.add_conv2d(silu1, conv2_w, Some(conv2_b), 1, 1, 1, 1, &feat_shape);
    let bn2_out = b.add_batch_norm(
        conv2_out,
        bn2_mean,
        bn2_var,
        bn2_weight,
        bn2_bias,
        bn2_eps,
        &feat_shape,
    );
    let sig2 = b.add_sigmoid(bn2_out, &feat_shape);
    let silu2 = b.add_binary_mul(bn2_out, sig2, &feat_shape);

    // Residual: output + input (shortcut=true)
    let out = b.add_binary_add(silu2, input, &feat_shape);

    b.build(out)
        .expect("valid DocLayout-YOLO Bottleneck residual kernel")
}

/// Bindings for Bottleneck residual block.
fn bottleneck_residual_bindings() -> Vec<TensorParamBinding> {
    let c = BOTTLENECK_CHANNELS;
    let conv_w = ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG);
    let conv_b = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_mean = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_weight = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_bias = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                          // features
        TensorParamBinding::ConstantTensor(conv_w.clone()),    // conv1_weight
        TensorParamBinding::ConstantTensor(conv_b.clone()),    // conv1_bias
        TensorParamBinding::ConstantTensor(bn_mean.clone()),   // bn1_running_mean
        TensorParamBinding::ConstantTensor(bn_var.clone()),    // bn1_running_var
        TensorParamBinding::ConstantTensor(bn_weight.clone()), // bn1_weight
        TensorParamBinding::ConstantTensor(bn_bias.clone()),   // bn1_bias
        TensorParamBinding::ConstantScalar(1e-5),              // bn1_eps
        TensorParamBinding::ConstantTensor(conv_w),            // conv2_weight
        TensorParamBinding::ConstantTensor(conv_b),            // conv2_bias
        TensorParamBinding::ConstantTensor(bn_mean),           // bn2_running_mean
        TensorParamBinding::ConstantTensor(bn_var),            // bn2_running_var
        TensorParamBinding::ConstantTensor(bn_weight),         // bn2_weight
        TensorParamBinding::ConstantTensor(bn_bias),           // bn2_bias
        TensorParamBinding::ConstantScalar(1e-5),              // bn2_eps
    ]
}

/// IBP bounds propagate through Bottleneck with residual skip connection.
///
/// The residual adds input to the Conv->BN->SiLU->Conv->BN->SiLU output.
/// IBP bounds on the sum should be wider than either branch alone.
#[test]
fn test_bottleneck_residual_ibp() {
    let def = build_bottleneck_residual_kernel();
    let bindings = bottleneck_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        2.0,
    );

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Bottleneck residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        "Bottleneck output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO Bottleneck residual IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds propagate through Bottleneck with residual skip.
///
/// SiLU (decomposed sigmoid + mul) requires CROWN linearization.
/// The residual skip is linear, so CROWN should handle it naturally.
#[test]
fn test_bottleneck_residual_crown() {
    let def = build_bottleneck_residual_kernel();
    let bindings = bottleneck_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        2.0,
    );

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO Bottleneck residual: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 6. C2f block: Conv split -> Bottleneck blocks -> channel concat -> Conv
// ===========================================================================

/// C2f intermediate channels (half of BOTTLENECK_CHANNELS for split).
const C2F_HALF: usize = BOTTLENECK_CHANNELS / 2; // 8

/// Build a simplified C2f block kernel.
///
/// Input: `[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE]` (Variable).
/// Output: `[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE]`.
///
/// C2f architecture (YOLOv8):
///   1. Conv2d(C, C, 1, s=1, p=0) -> BN -> SiLU (channel mixing)
///   2. Split output along channel dim: first_half, second_half
///   3. second_half -> Bottleneck1 (Conv->BN->SiLU->Conv->BN->SiLU + skip)
///   4. Bottleneck1 output -> Bottleneck2 (same)
///   5. Concat(first_half, second_half, bn1_out, bn2_out) along channels
///   6. Conv2d(C*2, C, 1, s=1, p=0) -> BN -> SiLU (channel reduction)
///
/// Simplified: We model 1x1 conv -> 2 bottleneck paths -> concat -> 1x1 conv.
/// Using full channel width (no split) for each bottleneck to keep graph
/// structure representative while maintaining tractable verification size.
fn build_c2f_block_kernel() -> TensorKernelDef {
    let c = BOTTLENECK_CHANNELS;
    let s = BOTTLENECK_SIZE;
    let feat_shape = [c, s, s];
    let mut b = TensorBlockBuilder::new("doclayout_c2f_block");

    let input = b.add_input("features", &feat_shape);

    // Entry 1x1 conv: Conv2d(C, C, 1, s=1, p=0) -> BN -> SiLU
    let entry_w = b.add_input("entry_conv_weight", &[c, c, 1, 1]);
    let entry_b = b.add_input("entry_conv_bias", &[c]);
    let entry_bn_mean = b.add_input("entry_bn_mean", &[c]);
    let entry_bn_var = b.add_input("entry_bn_var", &[c]);
    let entry_bn_weight = b.add_input("entry_bn_weight", &[c]);
    let entry_bn_bias = b.add_input("entry_bn_bias", &[c]);
    let entry_bn_eps = b.add_input("entry_bn_eps", &[1]);

    let entry_conv = b.add_conv2d(input, entry_w, Some(entry_b), 1, 1, 0, 0, &feat_shape);
    let entry_bn = b.add_batch_norm(
        entry_conv,
        entry_bn_mean,
        entry_bn_var,
        entry_bn_weight,
        entry_bn_bias,
        entry_bn_eps,
        &feat_shape,
    );
    let entry_sig = b.add_sigmoid(entry_bn, &feat_shape);
    let entry_silu = b.add_binary_mul(entry_bn, entry_sig, &feat_shape);

    // Bottleneck path: Conv2d(C, C, 3, s=1, p=1) -> BN -> SiLU
    let bn_conv_w = b.add_input("bn_conv_weight", &[c, c, 3, 3]);
    let bn_conv_b = b.add_input("bn_conv_bias", &[c]);
    let bn_bn_mean = b.add_input("bn_bn_mean", &[c]);
    let bn_bn_var = b.add_input("bn_bn_var", &[c]);
    let bn_bn_weight = b.add_input("bn_bn_weight", &[c]);
    let bn_bn_bias = b.add_input("bn_bn_bias", &[c]);
    let bn_bn_eps = b.add_input("bn_bn_eps", &[1]);

    let bn_conv = b.add_conv2d(
        entry_silu,
        bn_conv_w,
        Some(bn_conv_b),
        1,
        1,
        1,
        1,
        &feat_shape,
    );
    let bn_bn = b.add_batch_norm(
        bn_conv,
        bn_bn_mean,
        bn_bn_var,
        bn_bn_weight,
        bn_bn_bias,
        bn_bn_eps,
        &feat_shape,
    );
    let bn_sig = b.add_sigmoid(bn_bn, &feat_shape);
    let bn_silu = b.add_binary_mul(bn_bn, bn_sig, &feat_shape);

    // Residual: bottleneck output + entry_silu (skip connection)
    let bn_residual = b.add_binary_add(bn_silu, entry_silu, &feat_shape);

    // Concat entry_silu and bn_residual along channel dim
    let concat_channels = c * 2;
    let concat_shape = [concat_channels, s, s];
    let concat_out = b.add_concat(&[entry_silu, bn_residual], 0, &concat_shape);

    // Exit 1x1 conv: Conv2d(C*2, C, 1, s=1, p=0) -> BN -> SiLU
    let exit_w = b.add_input("exit_conv_weight", &[c, concat_channels, 1, 1]);
    let exit_b = b.add_input("exit_conv_bias", &[c]);
    let exit_bn_mean = b.add_input("exit_bn_mean", &[c]);
    let exit_bn_var = b.add_input("exit_bn_var", &[c]);
    let exit_bn_weight = b.add_input("exit_bn_weight", &[c]);
    let exit_bn_bias = b.add_input("exit_bn_bias", &[c]);
    let exit_bn_eps = b.add_input("exit_bn_eps", &[1]);

    let exit_conv = b.add_conv2d(concat_out, exit_w, Some(exit_b), 1, 1, 0, 0, &feat_shape);
    let exit_bn = b.add_batch_norm(
        exit_conv,
        exit_bn_mean,
        exit_bn_var,
        exit_bn_weight,
        exit_bn_bias,
        exit_bn_eps,
        &feat_shape,
    );
    let exit_sig = b.add_sigmoid(exit_bn, &feat_shape);
    let out = b.add_binary_mul(exit_bn, exit_sig, &feat_shape);

    b.build(out).expect("valid DocLayout-YOLO C2f block kernel")
}

/// Bindings for C2f block.
fn c2f_block_bindings() -> Vec<TensorParamBinding> {
    let c = BOTTLENECK_CHANNELS;
    let concat_channels = c * 2;

    // Helper closures for repetitive weight tensors
    let conv1x1_w =
        |c_in: usize, c_out: usize| ArrayD::from_elem(IxDyn(&[c_out, c_in, 1, 1]), WEIGHT_MAG);
    let conv3x3_w = || ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG);
    let zeros_c = || ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let ones_c = || ArrayD::from_elem(IxDyn(&[c]), 1.0f32);

    vec![
        TensorParamBinding::Variable, // features
        // Entry 1x1 conv
        TensorParamBinding::ConstantTensor(conv1x1_w(c, c)), // entry_conv_weight
        TensorParamBinding::ConstantTensor(zeros_c()),       // entry_conv_bias
        TensorParamBinding::ConstantTensor(zeros_c()),       // entry_bn_mean
        TensorParamBinding::ConstantTensor(ones_c()),        // entry_bn_var
        TensorParamBinding::ConstantTensor(ones_c()),        // entry_bn_weight
        TensorParamBinding::ConstantTensor(zeros_c()),       // entry_bn_bias
        TensorParamBinding::ConstantScalar(1e-5),            // entry_bn_eps
        // Bottleneck conv
        TensorParamBinding::ConstantTensor(conv3x3_w()), // bn_conv_weight
        TensorParamBinding::ConstantTensor(zeros_c()),   // bn_conv_bias
        TensorParamBinding::ConstantTensor(zeros_c()),   // bn_bn_mean
        TensorParamBinding::ConstantTensor(ones_c()),    // bn_bn_var
        TensorParamBinding::ConstantTensor(ones_c()),    // bn_bn_weight
        TensorParamBinding::ConstantTensor(zeros_c()),   // bn_bn_bias
        TensorParamBinding::ConstantScalar(1e-5),        // bn_bn_eps
        // Exit 1x1 conv
        TensorParamBinding::ConstantTensor(conv1x1_w(concat_channels, c)), // exit_conv_weight
        TensorParamBinding::ConstantTensor(zeros_c()),                     // exit_conv_bias
        TensorParamBinding::ConstantTensor(zeros_c()),                     // exit_bn_mean
        TensorParamBinding::ConstantTensor(ones_c()),                      // exit_bn_var
        TensorParamBinding::ConstantTensor(ones_c()),                      // exit_bn_weight
        TensorParamBinding::ConstantTensor(zeros_c()),                     // exit_bn_bias
        TensorParamBinding::ConstantScalar(1e-5),                          // exit_bn_eps
    ]
}

/// IBP bounds propagate through C2f block.
///
/// C2f is the core multi-branch block of YOLOv8/DocLayout-YOLO.
/// Tests: entry conv -> bottleneck with residual -> concat -> exit conv.
#[test]
fn test_c2f_block_ibp() {
    let def = build_c2f_block_kernel();
    let bindings = c2f_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        2.0,
    );

    let output = graph.propagate_ibp(&input).expect("IBP through C2f block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        "C2f block output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO C2f block IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds propagate through C2f block.
///
/// The C2f block has multiple SiLU nonlinearities and residual connections.
/// CROWN must linearize each sigmoid in the SiLU decomposition.
#[test]
fn test_c2f_block_crown() {
    let def = build_c2f_block_kernel();
    let bindings = c2f_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        2.0,
    );

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO C2f block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 7. PAN upsample + concat: Feature upsample + concat from backbone level
// ===========================================================================

/// PAN neck spatial sizes for two feature levels.
const PAN_HI_SIZE: usize = 8; // Higher resolution (P3-like)
const PAN_LO_SIZE: usize = 4; // Lower resolution (P4-like)
/// PAN channels for high-res and low-res features.
const PAN_HI_CHANNELS: usize = 16;
const PAN_LO_CHANNELS: usize = 32;

/// Build a PAN upsample + concat block.
///
/// Models the top-down path of the PAN neck: take a low-res feature map,
/// apply a 1x1 conv to match channel count, then concat with the
/// high-res feature map. Uses reshape to model the spatial dimension
/// change since TensorBlockBuilder does not have a native upsample op.
///
/// Input 1 (Variable): `[PAN_HI_CHANNELS, PAN_HI_SIZE, PAN_HI_SIZE]` (hi-res backbone features)
/// Input 2 (Variable): `[PAN_LO_CHANNELS, PAN_LO_SIZE, PAN_LO_SIZE]` (lo-res backbone features)
/// Output: `[PAN_HI_CHANNELS + UP_C, PAN_HI_SIZE, PAN_HI_SIZE]` (concatenated features)
///
/// Architecture:
///   lo_features -> Conv2d(LO_C, HI_C, 1, s=1, p=0) -> BN -> SiLU
///   -> Reshape to [UP_C, HI_SIZE, HI_SIZE] (models nearest upsample 2x)
///   concat(hi_features, upsampled_lo) along channel dim
///
/// A reshape preserves element count, so the `[HI_C, LO_SIZE, LO_SIZE]` conv
/// output cannot be reshaped to `[HI_C, HI_SIZE, HI_SIZE]` (4x the elements).
/// The nearest-neighbor 2x upsample is modeled soundly by trading channels for
/// spatial resolution: `[HI_C, LO_SIZE, LO_SIZE] -> [HI_C/4, HI_SIZE, HI_SIZE]`.
fn build_pan_upsample_concat_kernel() -> TensorKernelDef {
    let c_hi = PAN_HI_CHANNELS;
    let c_lo = PAN_LO_CHANNELS;
    let s_hi = PAN_HI_SIZE;
    let s_lo = PAN_LO_SIZE;
    // Reshape preserves element count: c_hi*s_lo*s_lo == up_c*s_hi*s_hi.
    let up_c = c_hi * s_lo * s_lo / (s_hi * s_hi);
    let hi_shape = [c_hi, s_hi, s_hi];
    let lo_shape = [c_lo, s_lo, s_lo];
    let conv_out_shape = [c_hi, s_lo, s_lo];
    let up_shape = [up_c, s_hi, s_hi];
    let out_channels = c_hi + up_c;
    let out_shape = [out_channels, s_hi, s_hi];
    let mut b = TensorBlockBuilder::new("doclayout_pan_upsample_concat");

    let hi_feat = b.add_input("hi_features", &hi_shape);
    let lo_feat = b.add_input("lo_features", &lo_shape);

    // 1x1 conv on lo-res to match hi-res channels
    let conv_w = b.add_input("conv_weight", &[c_hi, c_lo, 1, 1]);
    let conv_b = b.add_input("conv_bias", &[c_hi]);
    let bn_mean = b.add_input("bn_mean", &[c_hi]);
    let bn_var = b.add_input("bn_var", &[c_hi]);
    let bn_weight = b.add_input("bn_weight", &[c_hi]);
    let bn_bias = b.add_input("bn_bias", &[c_hi]);
    let bn_eps = b.add_input("bn_eps", &[1]);

    let conv_out = b.add_conv2d(lo_feat, conv_w, Some(conv_b), 1, 1, 0, 0, &conv_out_shape);
    let bn_out = b.add_batch_norm(
        conv_out,
        bn_mean,
        bn_var,
        bn_weight,
        bn_bias,
        bn_eps,
        &conv_out_shape,
    );
    let sig = b.add_sigmoid(bn_out, &conv_out_shape);
    let silu = b.add_binary_mul(bn_out, sig, &conv_out_shape);

    // Model upsample via reshape: [C_HI, S_LO, S_LO] -> [UP_C, S_HI, S_HI].
    // Element-count preserving (UP_C = C_HI/4): bounds propagation through
    // reshape preserves the element-wise bounds while changing the shape.
    let upsampled = b.add_reshape(silu, &up_shape);

    // Concat hi-res and upsampled lo-res along channel dim
    let out = b.add_concat(&[hi_feat, upsampled], 0, &out_shape);

    b.build(out)
        .expect("valid DocLayout-YOLO PAN upsample+concat kernel")
}

/// Bindings for PAN upsample + concat.
fn pan_upsample_concat_bindings() -> Vec<TensorParamBinding> {
    let c_hi = PAN_HI_CHANNELS;
    let c_lo = PAN_LO_CHANNELS;

    vec![
        TensorParamBinding::Variable, // hi_features
        TensorParamBinding::Variable, // lo_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c_hi, c_lo, 1, 1]),
            WEIGHT_MAG,
        )), // conv_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_hi]), 0.0f32)), // conv_bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_hi]), 0.0f32)), // bn_mean
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_hi]), 1.0f32)), // bn_var
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_hi]), 1.0f32)), // bn_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_hi]), 0.0f32)), // bn_bias
        TensorParamBinding::ConstantScalar(1e-5), // bn_eps
    ]
}

/// IBP bounds propagate through PAN upsample + concat.
///
/// Tests the top-down feature fusion path of the PAN neck. The 1x1 conv
/// changes channels, reshape models the upsample, and concat merges the
/// two feature levels.
#[test]
fn test_pan_upsample_concat_ibp() {
    let def = build_pan_upsample_concat_kernel();
    let bindings = pan_upsample_concat_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Multi-variable input: both feature maps are Variable
    let _hi_input = uniform_bounds(&[PAN_HI_CHANNELS, PAN_HI_SIZE, PAN_HI_SIZE], 2.0);
    // For multi-variable graphs, we create a single concatenated input
    let lo_flat_size = PAN_LO_CHANNELS * PAN_LO_SIZE * PAN_LO_SIZE;
    let hi_flat_size = PAN_HI_CHANNELS * PAN_HI_SIZE * PAN_HI_SIZE;
    let total_size = hi_flat_size + lo_flat_size;
    let input = uniform_bounds(&[total_size], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN upsample+concat");

    // Upsample reshape trades channels for spatial: up_c = c_hi/4, so the
    // concat over the channel dim yields c_hi + up_c channels.
    let up_c = PAN_HI_CHANNELS * PAN_LO_SIZE * PAN_LO_SIZE / (PAN_HI_SIZE * PAN_HI_SIZE);
    let out_channels = PAN_HI_CHANNELS + up_c;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[out_channels, PAN_HI_SIZE, PAN_HI_SIZE],
        "PAN upsample+concat output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO PAN upsample+concat IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. PAN downsample conv: ConvBnAct stride 2 -> concat
// ===========================================================================

/// Build a PAN downsample + concat block.
///
/// Models the bottom-up path: downsample hi-res features with stride-2
/// Conv2d, then concat with existing lo-res features.
///
/// Input 1 (Variable): `[PAN_HI_CHANNELS, PAN_HI_SIZE, PAN_HI_SIZE]` (hi-res features)
/// Input 2 (Variable): `[PAN_HI_CHANNELS, PAN_LO_SIZE, PAN_LO_SIZE]` (lo-res features)
/// Output: `[PAN_HI_CHANNELS * 2, PAN_LO_SIZE, PAN_LO_SIZE]` (concatenated)
fn build_pan_downsample_conv_kernel() -> TensorKernelDef {
    let c = PAN_HI_CHANNELS;
    let s_hi = PAN_HI_SIZE;
    let s_lo = PAN_LO_SIZE;
    let hi_shape = [c, s_hi, s_hi];
    let lo_shape = [c, s_lo, s_lo];
    let out_channels = c * 2;
    let out_shape = [out_channels, s_lo, s_lo];
    let mut b = TensorBlockBuilder::new("doclayout_pan_downsample_conv");

    let hi_feat = b.add_input("hi_features", &hi_shape);
    let lo_feat = b.add_input("lo_features", &lo_shape);

    // Stride-2 conv to downsample: [C, S_HI, S_HI] -> [C, S_LO, S_LO]
    let conv_w = b.add_input("conv_weight", &[c, c, 3, 3]);
    let conv_b = b.add_input("conv_bias", &[c]);
    let bn_mean = b.add_input("bn_mean", &[c]);
    let bn_var = b.add_input("bn_var", &[c]);
    let bn_weight = b.add_input("bn_weight", &[c]);
    let bn_bias = b.add_input("bn_bias", &[c]);
    let bn_eps = b.add_input("bn_eps", &[1]);

    let conv_out = b.add_conv2d(hi_feat, conv_w, Some(conv_b), 2, 2, 1, 1, &lo_shape);
    let bn_out = b.add_batch_norm(
        conv_out, bn_mean, bn_var, bn_weight, bn_bias, bn_eps, &lo_shape,
    );
    let sig = b.add_sigmoid(bn_out, &lo_shape);
    let downsampled = b.add_binary_mul(bn_out, sig, &lo_shape);

    // Concat downsampled hi-res with lo-res along channel dim
    let out = b.add_concat(&[downsampled, lo_feat], 0, &out_shape);

    b.build(out)
        .expect("valid DocLayout-YOLO PAN downsample+conv kernel")
}

/// Bindings for PAN downsample conv.
fn pan_downsample_conv_bindings() -> Vec<TensorParamBinding> {
    let c = PAN_HI_CHANNELS;

    vec![
        TensorParamBinding::Variable, // hi_features
        TensorParamBinding::Variable, // lo_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG)), // conv_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)), // conv_bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)), // bn_mean
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 1.0f32)), // bn_var
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 1.0f32)), // bn_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)), // bn_bias
        TensorParamBinding::ConstantScalar(1e-5),                                   // bn_eps
    ]
}

/// IBP bounds propagate through PAN downsample conv + concat.
///
/// Tests the bottom-up path: stride-2 conv downsamples spatial dims,
/// concat merges with existing lower-resolution features.
#[test]
fn test_pan_downsample_conv_ibp() {
    let def = build_pan_downsample_conv_kernel();
    let bindings = pan_downsample_conv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let hi_flat = PAN_HI_CHANNELS * PAN_HI_SIZE * PAN_HI_SIZE;
    let lo_flat = PAN_HI_CHANNELS * PAN_LO_SIZE * PAN_LO_SIZE;
    let input = uniform_bounds(&[hi_flat + lo_flat], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN downsample conv");

    let out_channels = PAN_HI_CHANNELS * 2;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[out_channels, PAN_LO_SIZE, PAN_LO_SIZE],
        "PAN downsample output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO PAN downsample conv IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Backbone two stages: ConvBnAct(3,16,3,2) -> ConvBnAct(16,32,3,2)
// ===========================================================================

/// Second-stage output channels.
const STAGE2_CHANNELS: usize = 32;
/// Spatial size after two stride-2 convolutions: 32/2/2 = 8.
const STAGE2_SIZE: usize = IMG_SIZE / 4; // 8

/// Build a 2-stage backbone kernel.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image [0, 1]).
/// Output: `[STAGE2_CHANNELS, STAGE2_SIZE, STAGE2_SIZE]`.
///
/// Stage 0: ConvBnAct(3, 16, k=3, s=2, p=1) -> SiLU
/// Stage 1: ConvBnAct(16, 32, k=3, s=2, p=1) -> SiLU
fn build_backbone_two_stages_kernel() -> TensorKernelDef {
    let c0 = CONV_OUT_CHANNELS; // 16
    let c1 = STAGE2_CHANNELS; // 32
    let s0 = CONV_OUT_SIZE; // 16
    let s1 = STAGE2_SIZE; // 8
    let shape0 = [c0, s0, s0];
    let shape1 = [c1, s1, s1];
    let mut b = TensorBlockBuilder::new("doclayout_backbone_two_stages");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Stage 0: Conv2d(3, 16, 3, s=2, p=1) -> BN -> SiLU
    let conv0_w = b.add_input("s0_conv_weight", &[c0, IN_CHANNELS, 3, 3]);
    let conv0_b = b.add_input("s0_conv_bias", &[c0]);
    let bn0_mean = b.add_input("s0_bn_mean", &[c0]);
    let bn0_var = b.add_input("s0_bn_var", &[c0]);
    let bn0_weight = b.add_input("s0_bn_weight", &[c0]);
    let bn0_bias = b.add_input("s0_bn_bias", &[c0]);
    let bn0_eps = b.add_input("s0_bn_eps", &[1]);

    let conv0 = b.add_conv2d(input, conv0_w, Some(conv0_b), 2, 2, 1, 1, &shape0);
    let bn0 = b.add_batch_norm(
        conv0, bn0_mean, bn0_var, bn0_weight, bn0_bias, bn0_eps, &shape0,
    );
    let sig0 = b.add_sigmoid(bn0, &shape0);
    let silu0 = b.add_binary_mul(bn0, sig0, &shape0);

    // Stage 1: Conv2d(16, 32, 3, s=2, p=1) -> BN -> SiLU
    let conv1_w = b.add_input("s1_conv_weight", &[c1, c0, 3, 3]);
    let conv1_b = b.add_input("s1_conv_bias", &[c1]);
    let bn1_mean = b.add_input("s1_bn_mean", &[c1]);
    let bn1_var = b.add_input("s1_bn_var", &[c1]);
    let bn1_weight = b.add_input("s1_bn_weight", &[c1]);
    let bn1_bias = b.add_input("s1_bn_bias", &[c1]);
    let bn1_eps = b.add_input("s1_bn_eps", &[1]);

    let conv1 = b.add_conv2d(silu0, conv1_w, Some(conv1_b), 2, 2, 1, 1, &shape1);
    let bn1 = b.add_batch_norm(
        conv1, bn1_mean, bn1_var, bn1_weight, bn1_bias, bn1_eps, &shape1,
    );
    let sig1 = b.add_sigmoid(bn1, &shape1);
    let out = b.add_binary_mul(bn1, sig1, &shape1);

    b.build(out)
        .expect("valid DocLayout-YOLO 2-stage backbone kernel")
}

/// Bindings for 2-stage backbone.
fn backbone_two_stages_bindings() -> Vec<TensorParamBinding> {
    let c0 = CONV_OUT_CHANNELS;
    let c1 = STAGE2_CHANNELS;

    vec![
        TensorParamBinding::Variable, // image
        // Stage 0
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c0, IN_CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // Stage 1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c1, c0, 3, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c1]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c1]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c1]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c1]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c1]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

/// IBP bounds through 2-stage backbone: image -> ConvBnAct -> ConvBnAct.
///
/// Two stride-2 convolutions downsample 32x32 -> 16x16 -> 8x8 while
/// expanding channels 3 -> 16 -> 32. Each stage has BN + SiLU nonlinearity.
#[test]
fn test_backbone_two_stages_ibp() {
    let def = build_backbone_two_stages_kernel();
    let bindings = backbone_two_stages_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-stage backbone");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[STAGE2_CHANNELS, STAGE2_SIZE, STAGE2_SIZE],
        "2-stage backbone output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO 2-stage backbone IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. Backbone full 5 stages: progressive channel expansion pyramid
// ===========================================================================

/// Channel widths for 5-stage backbone (matching DocLayoutYoloConfig default).
const BB_CHANNELS: [usize; 5] = [16, 32, 64, 128, 256];
/// Spatial sizes after each stride-2 conv: 32 -> 16 -> 8 -> 4 -> 2 -> 1.
const BB_SIZES: [usize; 5] = [
    IMG_SIZE / 2,  // 16
    IMG_SIZE / 4,  // 8
    IMG_SIZE / 8,  // 4
    IMG_SIZE / 16, // 2
    IMG_SIZE / 32, // 1
];

/// Build a 5-stage ConvBnAct pyramid (backbone without C2f/SPPF).
///
/// Input: `[3, 32, 32]` image (Variable).
/// Output: `[256, 1, 1]` deepest feature map.
///
/// Each stage: Conv2d(C_in, C_out, 3, s=2, p=1) -> BN -> SiLU.
fn build_backbone_full_five_stages_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("doclayout_backbone_five_stages");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Build 5 stages, each with ConvBnAct
    let mut prev = input;
    let mut prev_channels = IN_CHANNELS;

    for stage in 0..5 {
        let c_out = BB_CHANNELS[stage];
        let s_out = BB_SIZES[stage];
        let out_shape = [c_out, s_out, s_out];
        let prefix = format!("s{stage}");

        let conv_w = b.add_input(&format!("{prefix}_conv_w"), &[c_out, prev_channels, 3, 3]);
        let conv_b = b.add_input(&format!("{prefix}_conv_b"), &[c_out]);
        let bn_mean = b.add_input(&format!("{prefix}_bn_mean"), &[c_out]);
        let bn_var = b.add_input(&format!("{prefix}_bn_var"), &[c_out]);
        let bn_weight = b.add_input(&format!("{prefix}_bn_weight"), &[c_out]);
        let bn_bias = b.add_input(&format!("{prefix}_bn_bias"), &[c_out]);
        let bn_eps = b.add_input(&format!("{prefix}_bn_eps"), &[1]);

        let conv = b.add_conv2d(prev, conv_w, Some(conv_b), 2, 2, 1, 1, &out_shape);
        let bn = b.add_batch_norm(
            conv, bn_mean, bn_var, bn_weight, bn_bias, bn_eps, &out_shape,
        );
        let sig = b.add_sigmoid(bn, &out_shape);
        let silu = b.add_binary_mul(bn, sig, &out_shape);

        prev = silu;
        prev_channels = c_out;
    }

    b.build(prev)
        .expect("valid DocLayout-YOLO 5-stage backbone kernel")
}

/// Bindings for 5-stage backbone.
fn backbone_full_five_stages_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // image
    let mut prev_channels = IN_CHANNELS;

    for &c_out in &BB_CHANNELS {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c_out, prev_channels, 3, 3]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c_out]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c_out]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c_out]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c_out]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c_out]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        prev_channels = c_out;
    }

    bindings
}

/// IBP bounds through full 5-stage backbone pyramid.
///
/// Progressive spatial reduction 32x32 -> 16 -> 8 -> 4 -> 2 -> 1 with
/// channel expansion 3 -> 16 -> 32 -> 64 -> 128 -> 256. This tests deep
/// sequential IBP propagation through 5 ConvBnAct blocks.
#[test]
fn test_backbone_full_five_stages_ibp() {
    let def = build_backbone_full_five_stages_kernel();
    let bindings = backbone_full_five_stages_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 5-stage backbone");

    let final_c = BB_CHANNELS[4];
    let final_s = BB_SIZES[4];
    assert_eq!(
        output.lower_upper().0.shape(),
        &[final_c, final_s, final_s],
        "5-stage backbone output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO 5-stage backbone IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. Detect head box branch: Conv -> Conv -> DFL decode
// ===========================================================================

/// Detection head hidden channels.
const HEAD_HIDDEN: usize = 64;
/// Box regression output: 4 coordinates * DFL_BINS each.
const BOX_REG_OUT: usize = 4 * DFL_BINS; // 64

/// Build the box regression branch of the detection head.
///
/// Input: `[SPPF_CHANNELS, SPPF_SIZE, SPPF_SIZE]` (Variable, neck features).
/// Output: `[NUM_ANCHORS, 1]` after flattening + DFL decode.
///
/// Architecture:
///   Conv2d(C, HEAD_HIDDEN, 3, s=1, p=1) -> BN -> SiLU
///   Conv2d(HEAD_HIDDEN, BOX_REG_OUT, 1, s=1, p=0)
///   Reshape to [NUM_ANCHORS, 4, DFL_BINS]
///   Reshape to [NUM_ANCHORS * 4, DFL_BINS] (flatten for softmax)
///   Softmax(dim=1)
///   Matmul with DFL bins -> [NUM_ANCHORS * 4, 1]
fn build_detect_head_box_branch_kernel() -> TensorKernelDef {
    let c = SPPF_CHANNELS; // 64
    let s = SPPF_SIZE; // 8
    let feat_shape = [c, s, s];
    let hidden_shape = [HEAD_HIDDEN, s, s];
    let reg_shape = [BOX_REG_OUT, s, s];
    // NUM_ANCHORS = s * s = 64, flattened for DFL decode
    let flat_anchors = s * s; // 64
    let dfl_input_shape = [flat_anchors * 4, DFL_BINS];
    let dfl_output_shape = [flat_anchors * 4, 1];
    let mut b = TensorBlockBuilder::new("doclayout_detect_box_branch");

    let input = b.add_input("features", &feat_shape);

    // Conv2d(C, HEAD_HIDDEN, 3, s=1, p=1) -> BN -> SiLU
    let conv1_w = b.add_input("box_conv1_weight", &[HEAD_HIDDEN, c, 3, 3]);
    let conv1_b = b.add_input("box_conv1_bias", &[HEAD_HIDDEN]);
    let bn1_mean = b.add_input("box_bn1_mean", &[HEAD_HIDDEN]);
    let bn1_var = b.add_input("box_bn1_var", &[HEAD_HIDDEN]);
    let bn1_weight = b.add_input("box_bn1_weight", &[HEAD_HIDDEN]);
    let bn1_bias = b.add_input("box_bn1_bias", &[HEAD_HIDDEN]);
    let bn1_eps = b.add_input("box_bn1_eps", &[1]);

    let conv1 = b.add_conv2d(input, conv1_w, Some(conv1_b), 1, 1, 1, 1, &hidden_shape);
    let bn1 = b.add_batch_norm(
        conv1,
        bn1_mean,
        bn1_var,
        bn1_weight,
        bn1_bias,
        bn1_eps,
        &hidden_shape,
    );
    let sig1 = b.add_sigmoid(bn1, &hidden_shape);
    let silu1 = b.add_binary_mul(bn1, sig1, &hidden_shape);

    // Conv2d(HEAD_HIDDEN, BOX_REG_OUT, 1, s=1, p=0) — no BN/activation
    let conv2_w = b.add_input("box_conv2_weight", &[BOX_REG_OUT, HEAD_HIDDEN, 1, 1]);
    let conv2_b = b.add_input("box_conv2_bias", &[BOX_REG_OUT]);

    let conv2 = b.add_conv2d(silu1, conv2_w, Some(conv2_b), 1, 1, 0, 0, &reg_shape);

    // Reshape [BOX_REG_OUT, s, s] -> [flat_anchors * 4, DFL_BINS]
    let reshaped = b.add_reshape(conv2, &dfl_input_shape);

    // DFL decode: softmax + matmul with bins
    let bins = b.add_input("dfl_bins", &[DFL_BINS, 1]);
    let probs = b.add_softmax(reshaped, 1, &dfl_input_shape);
    let out = b.add_matmul(probs, bins, false, None, &dfl_output_shape);

    b.build(out)
        .expect("valid DocLayout-YOLO detect head box branch kernel")
}

/// Bindings for detect head box branch.
fn detect_head_box_branch_bindings() -> Vec<TensorParamBinding> {
    let c = SPPF_CHANNELS;
    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();

    vec![
        TensorParamBinding::Variable, // features
        // box conv1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HEAD_HIDDEN, c, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // box conv2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BOX_REG_OUT, HEAD_HIDDEN, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[BOX_REG_OUT]), 0.0f32)),
        // DFL bins
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins"),
        ),
    ]
}

/// IBP bounds through detect head box branch: Conv -> Conv -> DFL.
///
/// The DFL decode (softmax -> matmul with bins) converts logits to
/// continuous box coordinates. Softmax ensures the output is a weighted
/// sum of bin positions [0, ..., DFL_BINS-1].
#[test]
fn test_detect_head_box_branch_ibp() {
    let def = build_detect_head_box_branch_kernel();
    let bindings = detect_head_box_branch_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SPPF_CHANNELS, SPPF_SIZE, SPPF_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detect head box branch");

    let flat_anchors = SPPF_SIZE * SPPF_SIZE;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[flat_anchors * 4, 1],
        "detect head box branch output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO detect head box branch IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 12. Detect head cls branch: Conv -> Conv -> Sigmoid
// ===========================================================================

/// Build the classification branch of the detection head.
///
/// Input: `[SPPF_CHANNELS, SPPF_SIZE, SPPF_SIZE]` (Variable, neck features).
/// Output: `[NUM_ANCHORS, NUM_CLASSES]` class probabilities in [0, 1].
///
/// Architecture:
///   Conv2d(C, HEAD_HIDDEN, 3, s=1, p=1) -> BN -> SiLU
///   Conv2d(HEAD_HIDDEN, NUM_CLASSES, 1, s=1, p=0)
///   Reshape to [NUM_ANCHORS, NUM_CLASSES]
///   Sigmoid
fn build_detect_head_cls_branch_kernel() -> TensorKernelDef {
    let c = SPPF_CHANNELS; // 64
    let s = SPPF_SIZE; // 8
    let feat_shape = [c, s, s];
    let hidden_shape = [HEAD_HIDDEN, s, s];
    let cls_conv_shape = [NUM_CLASSES, s, s];
    let flat_anchors = s * s;
    let cls_output_shape = [flat_anchors, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("doclayout_detect_cls_branch");

    let input = b.add_input("features", &feat_shape);

    // Conv2d(C, HEAD_HIDDEN, 3, s=1, p=1) -> BN -> SiLU
    let conv1_w = b.add_input("cls_conv1_weight", &[HEAD_HIDDEN, c, 3, 3]);
    let conv1_b = b.add_input("cls_conv1_bias", &[HEAD_HIDDEN]);
    let bn1_mean = b.add_input("cls_bn1_mean", &[HEAD_HIDDEN]);
    let bn1_var = b.add_input("cls_bn1_var", &[HEAD_HIDDEN]);
    let bn1_weight = b.add_input("cls_bn1_weight", &[HEAD_HIDDEN]);
    let bn1_bias = b.add_input("cls_bn1_bias", &[HEAD_HIDDEN]);
    let bn1_eps = b.add_input("cls_bn1_eps", &[1]);

    let conv1 = b.add_conv2d(input, conv1_w, Some(conv1_b), 1, 1, 1, 1, &hidden_shape);
    let bn1 = b.add_batch_norm(
        conv1,
        bn1_mean,
        bn1_var,
        bn1_weight,
        bn1_bias,
        bn1_eps,
        &hidden_shape,
    );
    let sig1 = b.add_sigmoid(bn1, &hidden_shape);
    let silu1 = b.add_binary_mul(bn1, sig1, &hidden_shape);

    // Conv2d(HEAD_HIDDEN, NUM_CLASSES, 1, s=1, p=0) — no BN/activation
    let conv2_w = b.add_input("cls_conv2_weight", &[NUM_CLASSES, HEAD_HIDDEN, 1, 1]);
    let conv2_b = b.add_input("cls_conv2_bias", &[NUM_CLASSES]);

    let conv2 = b.add_conv2d(silu1, conv2_w, Some(conv2_b), 1, 1, 0, 0, &cls_conv_shape);

    // Reshape [NUM_CLASSES, s, s] -> [flat_anchors, NUM_CLASSES]
    let reshaped = b.add_reshape(conv2, &cls_output_shape);

    // Sigmoid: class probabilities in [0, 1]
    let out = b.add_sigmoid(reshaped, &cls_output_shape);

    b.build(out)
        .expect("valid DocLayout-YOLO detect head cls branch kernel")
}

/// Bindings for detect head cls branch.
fn detect_head_cls_branch_bindings() -> Vec<TensorParamBinding> {
    let c = SPPF_CHANNELS;

    vec![
        TensorParamBinding::Variable, // features
        // cls conv1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HEAD_HIDDEN, c, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // cls conv2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, HEAD_HIDDEN, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
    ]
}

/// IBP bounds through detect head cls branch: Conv -> Conv -> Sigmoid.
///
/// The final sigmoid ensures class probabilities are in [0, 1].
/// This is the key verification property for the classification head.
#[test]
fn test_detect_head_cls_branch_ibp() {
    let def = build_detect_head_cls_branch_kernel();
    let bindings = detect_head_cls_branch_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SPPF_CHANNELS, SPPF_SIZE, SPPF_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detect head cls branch");

    let flat_anchors = SPPF_SIZE * SPPF_SIZE;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[flat_anchors, NUM_CLASSES],
        "detect head cls branch output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO detect head cls branch IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid codomain is (0, 1).
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 13. Detect head full: box + cls parallel branches combined
// ===========================================================================

/// Build the full detection head with parallel box and cls branches.
///
/// Input: `[SPPF_CHANNELS, SPPF_SIZE, SPPF_SIZE]` (Variable, neck features).
/// Output: `[NUM_ANCHORS, NUM_CLASSES + 1]` (cls probs + 1 box coord dim).
///
/// The box branch produces DFL-decoded coordinates and the cls branch
/// produces sigmoid probabilities. Both share the same input features.
/// We combine a single DFL coordinate with all class scores via concat
/// to test parallel-branch bounds propagation.
fn build_detect_head_full_kernel() -> TensorKernelDef {
    let c = SPPF_CHANNELS;
    let s = SPPF_SIZE;
    let feat_shape = [c, s, s];
    let hidden_shape = [HEAD_HIDDEN, s, s];
    let flat_anchors = s * s;

    // Classification branch shapes
    let cls_conv_shape = [NUM_CLASSES, s, s];
    let cls_flat_shape = [flat_anchors, NUM_CLASSES];

    // Box regression branch shapes (single coordinate for simplicity)
    let box_conv_shape = [DFL_BINS, s, s];
    let box_flat_shape = [flat_anchors, DFL_BINS];
    let box_dfl_out = [flat_anchors, 1];

    // Combined output: [flat_anchors, NUM_CLASSES + 1]
    let combined_shape = [flat_anchors, NUM_CLASSES + 1];

    let mut b = TensorBlockBuilder::new("doclayout_detect_head_full");

    let input = b.add_input("features", &feat_shape);

    // -- Classification branch --
    let cls_conv1_w = b.add_input("cls_conv1_w", &[HEAD_HIDDEN, c, 3, 3]);
    let cls_conv1_b = b.add_input("cls_conv1_b", &[HEAD_HIDDEN]);
    let cls_bn_mean = b.add_input("cls_bn_mean", &[HEAD_HIDDEN]);
    let cls_bn_var = b.add_input("cls_bn_var", &[HEAD_HIDDEN]);
    let cls_bn_weight = b.add_input("cls_bn_weight", &[HEAD_HIDDEN]);
    let cls_bn_bias = b.add_input("cls_bn_bias", &[HEAD_HIDDEN]);
    let cls_bn_eps = b.add_input("cls_bn_eps", &[1]);

    let cls_c1 = b.add_conv2d(
        input,
        cls_conv1_w,
        Some(cls_conv1_b),
        1,
        1,
        1,
        1,
        &hidden_shape,
    );
    let cls_bn1 = b.add_batch_norm(
        cls_c1,
        cls_bn_mean,
        cls_bn_var,
        cls_bn_weight,
        cls_bn_bias,
        cls_bn_eps,
        &hidden_shape,
    );
    let cls_sig1 = b.add_sigmoid(cls_bn1, &hidden_shape);
    let cls_silu1 = b.add_binary_mul(cls_bn1, cls_sig1, &hidden_shape);

    let cls_conv2_w = b.add_input("cls_conv2_w", &[NUM_CLASSES, HEAD_HIDDEN, 1, 1]);
    let cls_conv2_b = b.add_input("cls_conv2_b", &[NUM_CLASSES]);
    let cls_c2 = b.add_conv2d(
        cls_silu1,
        cls_conv2_w,
        Some(cls_conv2_b),
        1,
        1,
        0,
        0,
        &cls_conv_shape,
    );
    let cls_reshaped = b.add_reshape(cls_c2, &cls_flat_shape);
    let cls_probs = b.add_sigmoid(cls_reshaped, &cls_flat_shape);

    // -- Box regression branch (single coordinate) --
    let box_conv1_w = b.add_input("box_conv1_w", &[HEAD_HIDDEN, c, 3, 3]);
    let box_conv1_b = b.add_input("box_conv1_b", &[HEAD_HIDDEN]);
    let box_bn_mean = b.add_input("box_bn_mean", &[HEAD_HIDDEN]);
    let box_bn_var = b.add_input("box_bn_var", &[HEAD_HIDDEN]);
    let box_bn_weight = b.add_input("box_bn_weight", &[HEAD_HIDDEN]);
    let box_bn_bias = b.add_input("box_bn_bias", &[HEAD_HIDDEN]);
    let box_bn_eps = b.add_input("box_bn_eps", &[1]);

    let box_c1 = b.add_conv2d(
        input,
        box_conv1_w,
        Some(box_conv1_b),
        1,
        1,
        1,
        1,
        &hidden_shape,
    );
    let box_bn1 = b.add_batch_norm(
        box_c1,
        box_bn_mean,
        box_bn_var,
        box_bn_weight,
        box_bn_bias,
        box_bn_eps,
        &hidden_shape,
    );
    let box_sig1 = b.add_sigmoid(box_bn1, &hidden_shape);
    let box_silu1 = b.add_binary_mul(box_bn1, box_sig1, &hidden_shape);

    let box_conv2_w = b.add_input("box_conv2_w", &[DFL_BINS, HEAD_HIDDEN, 1, 1]);
    let box_conv2_b = b.add_input("box_conv2_b", &[DFL_BINS]);
    let box_c2 = b.add_conv2d(
        box_silu1,
        box_conv2_w,
        Some(box_conv2_b),
        1,
        1,
        0,
        0,
        &box_conv_shape,
    );
    let box_reshaped = b.add_reshape(box_c2, &box_flat_shape);

    // DFL decode
    let bins = b.add_input("dfl_bins", &[DFL_BINS, 1]);
    let box_probs = b.add_softmax(box_reshaped, 1, &box_flat_shape);
    let box_coords = b.add_matmul(box_probs, bins, false, None, &box_dfl_out);

    // Combine: concat cls_probs and box_coords along last dim
    let out = b.add_concat(&[cls_probs, box_coords], 1, &combined_shape);

    b.build(out)
        .expect("valid DocLayout-YOLO full detect head kernel")
}

/// Bindings for full detect head.
fn detect_head_full_bindings() -> Vec<TensorParamBinding> {
    let c = SPPF_CHANNELS;
    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();

    vec![
        TensorParamBinding::Variable, // features
        // cls conv1 + BN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HEAD_HIDDEN, c, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // cls conv2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, HEAD_HIDDEN, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
        // box conv1 + BN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HEAD_HIDDEN, c, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_HIDDEN]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // box conv2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DFL_BINS, HEAD_HIDDEN, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DFL_BINS]), 0.0f32)),
        // DFL bins
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins"),
        ),
    ]
}

/// IBP bounds through full detect head: parallel box + cls branches.
///
/// Tests that both branches share the same input and produce valid bounds:
/// - Classification probabilities in [0, 1] (sigmoid)
/// - Box coordinates bounded by DFL decode (softmax + weighted sum)
#[test]
fn test_detect_head_full_ibp() {
    let def = build_detect_head_full_kernel();
    let bindings = detect_head_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SPPF_CHANNELS, SPPF_SIZE, SPPF_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full detect head");

    let flat_anchors = SPPF_SIZE * SPPF_SIZE;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[flat_anchors, NUM_CLASSES + 1],
        "full detect head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO full detect head IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 14. End-to-end detection: backbone 2-stage -> SPPF -> detect head
// ===========================================================================

/// Build an end-to-end detection pipeline (simplified).
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image [0, 1]).
/// Output: `[E2E_ANCHORS, NUM_CLASSES + 1]` (cls probs + 1 box coord).
///
/// Architecture:
///   Stage 0: ConvBnAct(3, 16, 3, s=2, p=1) -> SiLU    [16, 16, 16]
///   Stage 1: ConvBnAct(16, 64, 3, s=2, p=1) -> SiLU   [64, 8, 8]
///   SPPF: MaxPool chain + concat                        [256, 8, 8]
///   Cls head: Conv -> Conv -> Sigmoid                   [64, 10]
///   Box head: Conv -> Conv -> DFL                       [64, 1]
///   Concat: [64, 11]
///
/// Uses SPPF_CHANNELS=64 for the second stage to feed directly into SPPF.
const E2E_STAGE1_CHANNELS: usize = SPPF_CHANNELS; // 64
const E2E_STAGE1_SIZE: usize = SPPF_SIZE; // 8
const E2E_SPPF_OUT_CHANNELS: usize = SPPF_CHANNELS * 4; // 256
const E2E_ANCHORS: usize = E2E_STAGE1_SIZE * E2E_STAGE1_SIZE; // 64

fn build_end_to_end_detection_kernel() -> TensorKernelDef {
    let s0 = CONV_OUT_SIZE; // 16
    let c0 = CONV_OUT_CHANNELS; // 16
    let c1 = E2E_STAGE1_CHANNELS; // 64
    let s1 = E2E_STAGE1_SIZE; // 8
    let sppf_out_c = E2E_SPPF_OUT_CHANNELS; // 256

    let shape0 = [c0, s0, s0];
    let shape1 = [c1, s1, s1];
    let sppf_feat_shape = [c1, s1, s1];
    let sppf_out_shape = [sppf_out_c, s1, s1];

    let mut b = TensorBlockBuilder::new("doclayout_end_to_end_detection");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // -- Stage 0: ConvBnAct(3, 16, 3, s=2, p=1) -> SiLU --
    let s0_conv_w = b.add_input("s0_conv_w", &[c0, IN_CHANNELS, 3, 3]);
    let s0_conv_b = b.add_input("s0_conv_b", &[c0]);
    let s0_bn_mean = b.add_input("s0_bn_mean", &[c0]);
    let s0_bn_var = b.add_input("s0_bn_var", &[c0]);
    let s0_bn_weight = b.add_input("s0_bn_weight", &[c0]);
    let s0_bn_bias = b.add_input("s0_bn_bias", &[c0]);
    let s0_bn_eps = b.add_input("s0_bn_eps", &[1]);

    let s0_conv = b.add_conv2d(input, s0_conv_w, Some(s0_conv_b), 2, 2, 1, 1, &shape0);
    let s0_bn = b.add_batch_norm(
        s0_conv,
        s0_bn_mean,
        s0_bn_var,
        s0_bn_weight,
        s0_bn_bias,
        s0_bn_eps,
        &shape0,
    );
    let s0_sig = b.add_sigmoid(s0_bn, &shape0);
    let s0_silu = b.add_binary_mul(s0_bn, s0_sig, &shape0);

    // -- Stage 1: ConvBnAct(16, 64, 3, s=2, p=1) -> SiLU --
    let s1_conv_w = b.add_input("s1_conv_w", &[c1, c0, 3, 3]);
    let s1_conv_b = b.add_input("s1_conv_b", &[c1]);
    let s1_bn_mean = b.add_input("s1_bn_mean", &[c1]);
    let s1_bn_var = b.add_input("s1_bn_var", &[c1]);
    let s1_bn_weight = b.add_input("s1_bn_weight", &[c1]);
    let s1_bn_bias = b.add_input("s1_bn_bias", &[c1]);
    let s1_bn_eps = b.add_input("s1_bn_eps", &[1]);

    let s1_conv = b.add_conv2d(s0_silu, s1_conv_w, Some(s1_conv_b), 2, 2, 1, 1, &shape1);
    let s1_bn = b.add_batch_norm(
        s1_conv,
        s1_bn_mean,
        s1_bn_var,
        s1_bn_weight,
        s1_bn_bias,
        s1_bn_eps,
        &shape1,
    );
    let s1_sig = b.add_sigmoid(s1_bn, &shape1);
    let s1_silu = b.add_binary_mul(s1_bn, s1_sig, &shape1);

    // -- SPPF: 3x MaxPool2d(k=5, s=1, p=2) + concat --
    let pool1 = b.add_max_pool_2d(
        s1_silu,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &sppf_feat_shape,
    );
    let pool2 = b.add_max_pool_2d(
        pool1,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &sppf_feat_shape,
    );
    let pool3 = b.add_max_pool_2d(
        pool2,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &sppf_feat_shape,
    );
    let sppf_out = b.add_concat(&[s1_silu, pool1, pool2, pool3], 0, &sppf_out_shape);

    // -- 1x1 conv to reduce SPPF channels for detect head --
    let reduce_c = SPPF_CHANNELS; // 64
    let reduce_shape = [reduce_c, s1, s1];
    let reduce_w = b.add_input("reduce_conv_w", &[reduce_c, sppf_out_c, 1, 1]);
    let reduce_b = b.add_input("reduce_conv_b", &[reduce_c]);

    let reduce_conv = b.add_conv2d(
        sppf_out,
        reduce_w,
        Some(reduce_b),
        1,
        1,
        0,
        0,
        &reduce_shape,
    );

    // -- Cls head: Conv -> Reshape -> Sigmoid --
    let cls_w = b.add_input("cls_conv_w", &[NUM_CLASSES, reduce_c, 1, 1]);
    let cls_b = b.add_input("cls_conv_b", &[NUM_CLASSES]);
    let cls_conv_shape = [NUM_CLASSES, s1, s1];
    let cls_flat_shape = [E2E_ANCHORS, NUM_CLASSES];

    let cls_conv = b.add_conv2d(reduce_conv, cls_w, Some(cls_b), 1, 1, 0, 0, &cls_conv_shape);
    let cls_reshaped = b.add_reshape(cls_conv, &cls_flat_shape);
    let cls_probs = b.add_sigmoid(cls_reshaped, &cls_flat_shape);

    // -- Box head: Conv -> Reshape -> Softmax -> DFL --
    let box_w = b.add_input("box_conv_w", &[DFL_BINS, reduce_c, 1, 1]);
    let box_b = b.add_input("box_conv_b", &[DFL_BINS]);
    let box_conv_shape = [DFL_BINS, s1, s1];
    let box_flat_shape = [E2E_ANCHORS, DFL_BINS];
    let box_dfl_out_shape = [E2E_ANCHORS, 1];

    let box_conv = b.add_conv2d(reduce_conv, box_w, Some(box_b), 1, 1, 0, 0, &box_conv_shape);
    let box_reshaped = b.add_reshape(box_conv, &box_flat_shape);
    let bins = b.add_input("dfl_bins", &[DFL_BINS, 1]);
    let box_probs = b.add_softmax(box_reshaped, 1, &box_flat_shape);
    let box_coords = b.add_matmul(box_probs, bins, false, None, &box_dfl_out_shape);

    // -- Combine cls + box via concat --
    let combined_shape = [E2E_ANCHORS, NUM_CLASSES + 1];
    let out = b.add_concat(&[cls_probs, box_coords], 1, &combined_shape);

    b.build(out)
        .expect("valid DocLayout-YOLO end-to-end detection kernel")
}

/// Bindings for end-to-end detection.
fn end_to_end_detection_bindings() -> Vec<TensorParamBinding> {
    let c0 = CONV_OUT_CHANNELS;
    let c1 = E2E_STAGE1_CHANNELS;
    let sppf_out_c = E2E_SPPF_OUT_CHANNELS;
    let reduce_c = SPPF_CHANNELS;
    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();

    vec![
        TensorParamBinding::Variable, // image
        // Stage 0
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c0, IN_CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // Stage 1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c1, c0, 3, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c1]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c1]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c1]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c1]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c1]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // Reduce conv (1x1, no BN)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[reduce_c, sppf_out_c, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[reduce_c]), 0.0f32)),
        // Cls conv
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, reduce_c, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
        // Box conv
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DFL_BINS, reduce_c, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DFL_BINS]), 0.0f32)),
        // DFL bins
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins"),
        ),
    ]
}

/// IBP bounds through end-to-end detection: backbone -> SPPF -> detect head.
///
/// The most comprehensive test: verifies bounds propagation through the
/// complete DocLayout-YOLO pipeline from raw image input to detection outputs.
/// Tests deep sequential propagation (2 ConvBnAct stages), SPPF multi-scale
/// aggregation, and parallel detect head branches (cls sigmoid + box DFL).
#[test]
fn test_end_to_end_detection_ibp() {
    let def = build_end_to_end_detection_kernel();
    let bindings = end_to_end_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through end-to-end detection");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[E2E_ANCHORS, NUM_CLASSES + 1],
        "end-to-end detection output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO end-to-end detection IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 15. C2f 3-bottleneck chain: 3 sequential bottleneck residual blocks
// ===========================================================================

/// Build a C2f block with 3 sequential bottleneck residual blocks.
///
/// Input: `[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE]` (Variable).
/// Output: `[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE]`.
///
/// Architecture (YOLOv8 C2f with n=3):
///   Entry 1x1 conv -> BN -> SiLU
///   Bottleneck 1: Conv3x3 -> BN -> SiLU + residual
///   Bottleneck 2: Conv3x3 -> BN -> SiLU + residual
///   Bottleneck 3: Conv3x3 -> BN -> SiLU + residual
///   Concat(entry, bn1, bn2, bn3) along channels
///   Exit 1x1 conv -> BN -> SiLU (channel reduction)
///
/// Tests deep sequential bounds propagation through 3 residual branches.
fn build_c2f_three_bottleneck_kernel() -> TensorKernelDef {
    let c = BOTTLENECK_CHANNELS;
    let s = BOTTLENECK_SIZE;
    let feat_shape = [c, s, s];
    let mut b = TensorBlockBuilder::new("doclayout_c2f_3bn");

    let input = b.add_input("features", &feat_shape);

    // Entry 1x1 conv: Conv2d(C, C, 1, s=1, p=0) -> BN -> SiLU
    let entry_w = b.add_input("entry_w", &[c, c, 1, 1]);
    let entry_b = b.add_input("entry_b", &[c]);
    let entry_bn_mean = b.add_input("entry_bn_mean", &[c]);
    let entry_bn_var = b.add_input("entry_bn_var", &[c]);
    let entry_bn_w = b.add_input("entry_bn_w", &[c]);
    let entry_bn_b = b.add_input("entry_bn_b", &[c]);
    let entry_bn_eps = b.add_input("entry_bn_eps", &[1]);

    let entry_conv = b.add_conv2d(input, entry_w, Some(entry_b), 1, 1, 0, 0, &feat_shape);
    let entry_bn = b.add_batch_norm(
        entry_conv,
        entry_bn_mean,
        entry_bn_var,
        entry_bn_w,
        entry_bn_b,
        entry_bn_eps,
        &feat_shape,
    );
    let entry_sig = b.add_sigmoid(entry_bn, &feat_shape);
    let entry_silu = b.add_binary_mul(entry_bn, entry_sig, &feat_shape);

    // Helper: build one bottleneck residual block
    // Each bottleneck: Conv3x3 -> BN -> SiLU + skip
    let mut prev = entry_silu;
    let mut bottleneck_outputs = vec![entry_silu];

    for i in 0..3 {
        let prefix = format!("bn{i}");
        let cw = b.add_input(&format!("{prefix}_conv_w"), &[c, c, 3, 3]);
        let cb = b.add_input(&format!("{prefix}_conv_b"), &[c]);
        let bm = b.add_input(&format!("{prefix}_bn_mean"), &[c]);
        let bv = b.add_input(&format!("{prefix}_bn_var"), &[c]);
        let bw = b.add_input(&format!("{prefix}_bn_w"), &[c]);
        let bb = b.add_input(&format!("{prefix}_bn_b"), &[c]);
        let be = b.add_input(&format!("{prefix}_bn_eps"), &[1]);

        let conv = b.add_conv2d(prev, cw, Some(cb), 1, 1, 1, 1, &feat_shape);
        let bn = b.add_batch_norm(conv, bm, bv, bw, bb, be, &feat_shape);
        let sig = b.add_sigmoid(bn, &feat_shape);
        let silu = b.add_binary_mul(bn, sig, &feat_shape);
        let residual = b.add_binary_add(silu, prev, &feat_shape);

        bottleneck_outputs.push(residual);
        prev = residual;
    }

    // Concat entry + 3 bottleneck outputs along channel dim
    let concat_channels = c * 4;
    let concat_shape = [concat_channels, s, s];
    let concat_out = b.add_concat(&bottleneck_outputs, 0, &concat_shape);

    // Exit 1x1 conv: Conv2d(C*4, C, 1, s=1, p=0) -> BN -> SiLU
    let exit_w = b.add_input("exit_w", &[c, concat_channels, 1, 1]);
    let exit_b = b.add_input("exit_b", &[c]);
    let exit_bn_mean = b.add_input("exit_bn_mean", &[c]);
    let exit_bn_var = b.add_input("exit_bn_var", &[c]);
    let exit_bn_w = b.add_input("exit_bn_w", &[c]);
    let exit_bn_b = b.add_input("exit_bn_b", &[c]);
    let exit_bn_eps = b.add_input("exit_bn_eps", &[1]);

    let exit_conv = b.add_conv2d(concat_out, exit_w, Some(exit_b), 1, 1, 0, 0, &feat_shape);
    let exit_bn = b.add_batch_norm(
        exit_conv,
        exit_bn_mean,
        exit_bn_var,
        exit_bn_w,
        exit_bn_b,
        exit_bn_eps,
        &feat_shape,
    );
    let exit_sig = b.add_sigmoid(exit_bn, &feat_shape);
    let out = b.add_binary_mul(exit_bn, exit_sig, &feat_shape);

    b.build(out)
        .expect("valid DocLayout-YOLO C2f 3-bottleneck kernel")
}

/// Bindings for C2f 3-bottleneck.
fn c2f_three_bottleneck_bindings() -> Vec<TensorParamBinding> {
    let c = BOTTLENECK_CHANNELS;
    let concat_channels = c * 4;

    let conv1x1_w =
        |c_in: usize, c_out: usize| ArrayD::from_elem(IxDyn(&[c_out, c_in, 1, 1]), WEIGHT_MAG);
    let conv3x3_w = || ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG);
    let zeros_c = || ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let ones_c = || ArrayD::from_elem(IxDyn(&[c]), 1.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable, // features
        // Entry 1x1 conv
        TensorParamBinding::ConstantTensor(conv1x1_w(c, c)),
        TensorParamBinding::ConstantTensor(zeros_c()),
        TensorParamBinding::ConstantTensor(zeros_c()),
        TensorParamBinding::ConstantTensor(ones_c()),
        TensorParamBinding::ConstantTensor(ones_c()),
        TensorParamBinding::ConstantTensor(zeros_c()),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    // 3 bottleneck blocks
    for _ in 0..3 {
        bindings.push(TensorParamBinding::ConstantTensor(conv3x3_w()));
        bindings.push(TensorParamBinding::ConstantTensor(zeros_c()));
        bindings.push(TensorParamBinding::ConstantTensor(zeros_c()));
        bindings.push(TensorParamBinding::ConstantTensor(ones_c()));
        bindings.push(TensorParamBinding::ConstantTensor(ones_c()));
        bindings.push(TensorParamBinding::ConstantTensor(zeros_c()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    }

    // Exit 1x1 conv
    bindings.push(TensorParamBinding::ConstantTensor(conv1x1_w(
        concat_channels,
        c,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(zeros_c()));
    bindings.push(TensorParamBinding::ConstantTensor(zeros_c()));
    bindings.push(TensorParamBinding::ConstantTensor(ones_c()));
    bindings.push(TensorParamBinding::ConstantTensor(ones_c()));
    bindings.push(TensorParamBinding::ConstantTensor(zeros_c()));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    bindings
}

/// IBP bounds through C2f with 3 bottleneck residual blocks.
///
/// Deep sequential propagation through 3 residual branches, each with
/// Conv3x3 -> BN -> SiLU + skip, followed by 4-way channel concat and
/// exit 1x1 conv.
#[test]
fn test_c2f_three_bottleneck_ibp() {
    let def = build_c2f_three_bottleneck_kernel();
    let bindings = c2f_three_bottleneck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        2.0,
    );

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through C2f 3-bottleneck");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        "C2f 3-bottleneck output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO C2f 3-bottleneck IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through C2f 3-bottleneck chain.
///
/// 3 sequential SiLU nonlinearities plus residual additions produce a
/// deeply nonlinear graph. CROWN linearizes each sigmoid in the SiLU
/// decomposition across all 3 bottlenecks.
#[test]
fn test_c2f_three_bottleneck_crown() {
    let def = build_c2f_three_bottleneck_kernel();
    let bindings = c2f_three_bottleneck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        2.0,
    );

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO C2f 3-bottleneck: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 16. PAN neck 3-scale: upsample P5->P4->P3, then downsample P3->P4->P5
// ===========================================================================

/// PAN 3-scale channel widths.
const PAN3_P3_CHANNELS: usize = 8;
const PAN3_P4_CHANNELS: usize = 16;
const PAN3_P5_CHANNELS: usize = 32;
/// PAN 3-scale spatial sizes: P3=8, P4=4, P5=2.
const PAN3_P3_SIZE: usize = 8;
const PAN3_P4_SIZE: usize = 4;
const PAN3_P5_SIZE: usize = 2;

/// Build a simplified PAN 3-scale top-down path (upsample).
///
/// Models the top-down fusion of the PAN neck across 3 feature scales.
/// P5 is the deepest (smallest spatial), P3 is the shallowest (largest spatial).
///
/// Input (Variable): `[PAN3_P5_CHANNELS, PAN3_P5_SIZE, PAN3_P5_SIZE]` (P5 features).
/// Constant: P4-like features, P3-like features (simulated via constant bounds).
/// Output: `[PAN3_P3_CHANNELS + UP4_C, PAN3_P3_SIZE, PAN3_P3_SIZE]` after final concat.
///
/// Architecture (top-down):
///   P5 -> 1x1 conv (channel reduce) -> reshape (upsample 2x to P4 size)
///        concat with P4 features -> 1x1 conv -> BN -> SiLU
///   -> reshape (upsample 2x to P3 size) -> concat with P3 features
///
/// A reshape preserves element count, so each nearest-neighbor 2x upsample is
/// modeled by trading channels for spatial resolution (channels/4), not by
/// reshaping to a 4x-larger element count. Downstream channel counts follow.
fn build_pan_three_scale_topdown_kernel() -> TensorKernelDef {
    let c5 = PAN3_P5_CHANNELS;
    let c4 = PAN3_P4_CHANNELS;
    let c3 = PAN3_P3_CHANNELS;
    let s5 = PAN3_P5_SIZE;
    let s4 = PAN3_P4_SIZE;
    let s3 = PAN3_P3_SIZE;
    // 2x upsample reshapes preserve element count, so channels drop by 4x:
    //   up5_c = c4*s5*s5/(s4*s4), up4_c = c3*s4*s4/(s3*s3).
    let up5_c = c4 * s5 * s5 / (s4 * s4);
    let concat54_c = c4 + up5_c;

    let mut b = TensorBlockBuilder::new("doclayout_pan_3scale_topdown");

    // P5 features as variable input
    let p5 = b.add_input("p5_features", &[c5, s5, s5]);
    // P4 and P3 features as constant inputs (backbone lateral outputs)
    let p4_feat = b.add_input("p4_features", &[c4, s4, s4]);
    let p3_feat = b.add_input("p3_features", &[c3, s3, s3]);

    // P5 -> 1x1 conv to reduce channels to c4
    let reduce5_w = b.add_input("reduce5_w", &[c4, c5, 1, 1]);
    let reduce5_b = b.add_input("reduce5_b", &[c4]);
    let reduce5 = b.add_conv2d(p5, reduce5_w, Some(reduce5_b), 1, 1, 0, 0, &[c4, s5, s5]);

    // Reshape (model upsample 2x): [c4, s5, s5] -> [up5_c, s4, s4] (count-preserving)
    let up5_to_4 = b.add_reshape(reduce5, &[up5_c, s4, s4]);

    // Concat upsampled P5 with P4 along channels
    let concat54_shape = [concat54_c, s4, s4];
    let concat54 = b.add_concat(&[up5_to_4, p4_feat], 0, &concat54_shape);

    // 1x1 conv to merge: [concat54_c, s4, s4] -> [c3, s4, s4]
    let merge4_w = b.add_input("merge4_w", &[c3, concat54_c, 1, 1]);
    let merge4_b = b.add_input("merge4_b", &[c3]);
    let merge4_bn_mean = b.add_input("merge4_bn_mean", &[c3]);
    let merge4_bn_var = b.add_input("merge4_bn_var", &[c3]);
    let merge4_bn_w = b.add_input("merge4_bn_w", &[c3]);
    let merge4_bn_b = b.add_input("merge4_bn_b", &[c3]);
    let merge4_bn_eps = b.add_input("merge4_bn_eps", &[1]);

    let merge4_conv = b.add_conv2d(
        concat54,
        merge4_w,
        Some(merge4_b),
        1,
        1,
        0,
        0,
        &[c3, s4, s4],
    );
    let merge4_bn = b.add_batch_norm(
        merge4_conv,
        merge4_bn_mean,
        merge4_bn_var,
        merge4_bn_w,
        merge4_bn_b,
        merge4_bn_eps,
        &[c3, s4, s4],
    );
    let merge4_sig = b.add_sigmoid(merge4_bn, &[c3, s4, s4]);
    let merge4_silu = b.add_binary_mul(merge4_bn, merge4_sig, &[c3, s4, s4]);

    // Reshape (model upsample 2x): [c3, s4, s4] -> [up4_c, s3, s3] (count-preserving)
    let up4_c = c3 * s4 * s4 / (s3 * s3);
    let up4_to_3 = b.add_reshape(merge4_silu, &[up4_c, s3, s3]);

    // Concat upsampled merge4 with P3 along channels
    let out_channels = c3 + up4_c;
    let out_shape = [out_channels, s3, s3];
    let out = b.add_concat(&[up4_to_3, p3_feat], 0, &out_shape);

    b.build(out)
        .expect("valid DocLayout-YOLO PAN 3-scale top-down kernel")
}

/// Bindings for PAN 3-scale top-down.
fn pan_three_scale_topdown_bindings() -> Vec<TensorParamBinding> {
    let c5 = PAN3_P5_CHANNELS;
    let c4 = PAN3_P4_CHANNELS;
    let c3 = PAN3_P3_CHANNELS;
    let s5 = PAN3_P5_SIZE;
    let s4 = PAN3_P4_SIZE;
    let s3 = PAN3_P3_SIZE;
    // Mirror the count-preserving upsample reshape: up5_c = c4/4, so merge4
    // consumes c4 + up5_c channels.
    let up5_c = c4 * s5 * s5 / (s4 * s4);
    let concat54_c = c4 + up5_c;

    vec![
        TensorParamBinding::Variable, // p5_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4, s4, s4]), 0.5f32)), // p4_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c3, s3, s3]), 0.5f32)), // p3_features
        // reduce5 conv
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4, c5, 1, 1]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 0.0f32)),
        // merge4 conv + BN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c3, concat54_c, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c3]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c3]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c3]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c3]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c3]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

/// IBP bounds through PAN 3-scale top-down feature fusion.
///
/// Tests progressive upsample + concat + merge across 3 feature pyramid levels.
/// P5 (deepest, 2x2) -> merge with P4 (4x4) -> merge with P3 (8x8).
#[test]
fn test_pan_three_scale_topdown_ibp() {
    let def = build_pan_three_scale_topdown_kernel();
    let bindings = pan_three_scale_topdown_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[PAN3_P5_CHANNELS, PAN3_P5_SIZE, PAN3_P5_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN 3-scale top-down");

    // Final upsample reshape trades channels for spatial: up4_c = c3/4, so the
    // last concat yields c3 + up4_c channels.
    let up4_c = PAN3_P3_CHANNELS * PAN3_P4_SIZE * PAN3_P4_SIZE / (PAN3_P3_SIZE * PAN3_P3_SIZE);
    let out_channels = PAN3_P3_CHANNELS + up4_c;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[out_channels, PAN3_P3_SIZE, PAN3_P3_SIZE],
        "PAN 3-scale top-down output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO PAN 3-scale top-down IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 17. PAN bottom-up path: downsample P3->P4->P5
// ===========================================================================

/// Build a PAN bottom-up path: stride-2 downsample P3->P4, then P4->P5.
///
/// Input (Variable): `[PAN3_P3_CHANNELS, PAN3_P3_SIZE, PAN3_P3_SIZE]` (P3 features).
/// Constant: P4-like features, P5-like features.
/// Output: `[PAN3_P5_CHANNELS * 2, PAN3_P5_SIZE, PAN3_P5_SIZE]` (final concat).
fn build_pan_bottom_up_kernel() -> TensorKernelDef {
    let c3 = PAN3_P3_CHANNELS;
    let c4 = PAN3_P4_CHANNELS;
    let c5 = PAN3_P5_CHANNELS;
    let s3 = PAN3_P3_SIZE;
    let s4 = PAN3_P4_SIZE;
    let s5 = PAN3_P5_SIZE;

    let mut b = TensorBlockBuilder::new("doclayout_pan_bottom_up");

    let p3 = b.add_input("p3_features", &[c3, s3, s3]);
    let p4_feat = b.add_input("p4_features", &[c4, s4, s4]);
    let p5_feat = b.add_input("p5_features", &[c5, s5, s5]);

    // Downsample P3 -> P4 size: stride-2 conv
    let down34_w = b.add_input("down34_w", &[c4, c3, 3, 3]);
    let down34_b = b.add_input("down34_b", &[c4]);
    let down34_bn_mean = b.add_input("down34_bn_mean", &[c4]);
    let down34_bn_var = b.add_input("down34_bn_var", &[c4]);
    let down34_bn_w = b.add_input("down34_bn_w", &[c4]);
    let down34_bn_b = b.add_input("down34_bn_b", &[c4]);
    let down34_bn_eps = b.add_input("down34_bn_eps", &[1]);

    let down34 = b.add_conv2d(p3, down34_w, Some(down34_b), 2, 2, 1, 1, &[c4, s4, s4]);
    let down34_bn = b.add_batch_norm(
        down34,
        down34_bn_mean,
        down34_bn_var,
        down34_bn_w,
        down34_bn_b,
        down34_bn_eps,
        &[c4, s4, s4],
    );
    let down34_sig = b.add_sigmoid(down34_bn, &[c4, s4, s4]);
    let down34_silu = b.add_binary_mul(down34_bn, down34_sig, &[c4, s4, s4]);

    // Concat downsampled P3 with P4
    let concat34_shape = [c4 * 2, s4, s4];
    let concat34 = b.add_concat(&[down34_silu, p4_feat], 0, &concat34_shape);

    // Merge 1x1 conv: [c4*2, s4, s4] -> [c4, s4, s4]
    let merge4_w = b.add_input("merge4_w", &[c4, c4 * 2, 1, 1]);
    let merge4_b = b.add_input("merge4_b", &[c4]);
    let merge4 = b.add_conv2d(
        concat34,
        merge4_w,
        Some(merge4_b),
        1,
        1,
        0,
        0,
        &[c4, s4, s4],
    );

    // Downsample merged P4 -> P5 size: stride-2 conv
    let down45_w = b.add_input("down45_w", &[c5, c4, 3, 3]);
    let down45_b = b.add_input("down45_b", &[c5]);
    let down45_bn_mean = b.add_input("down45_bn_mean", &[c5]);
    let down45_bn_var = b.add_input("down45_bn_var", &[c5]);
    let down45_bn_w = b.add_input("down45_bn_w", &[c5]);
    let down45_bn_b = b.add_input("down45_bn_b", &[c5]);
    let down45_bn_eps = b.add_input("down45_bn_eps", &[1]);

    let down45 = b.add_conv2d(merge4, down45_w, Some(down45_b), 2, 2, 1, 1, &[c5, s5, s5]);
    let down45_bn = b.add_batch_norm(
        down45,
        down45_bn_mean,
        down45_bn_var,
        down45_bn_w,
        down45_bn_b,
        down45_bn_eps,
        &[c5, s5, s5],
    );
    let down45_sig = b.add_sigmoid(down45_bn, &[c5, s5, s5]);
    let down45_silu = b.add_binary_mul(down45_bn, down45_sig, &[c5, s5, s5]);

    // Concat downsampled with P5
    let out_channels = c5 * 2;
    let out_shape = [out_channels, s5, s5];
    let out = b.add_concat(&[down45_silu, p5_feat], 0, &out_shape);

    b.build(out)
        .expect("valid DocLayout-YOLO PAN bottom-up kernel")
}

/// Bindings for PAN bottom-up.
fn pan_bottom_up_bindings() -> Vec<TensorParamBinding> {
    let c3 = PAN3_P3_CHANNELS;
    let c4 = PAN3_P4_CHANNELS;
    let c5 = PAN3_P5_CHANNELS;
    let s4 = PAN3_P4_SIZE;
    let s5 = PAN3_P5_SIZE;

    vec![
        TensorParamBinding::Variable, // p3_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4, s4, s4]), 0.5f32)), // p4_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5, s5, s5]), 0.5f32)), // p5_features
        // down34 conv + BN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4, c3, 3, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // merge4 conv
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c4, c4 * 2, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 0.0f32)),
        // down45 conv + BN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5, c4, 3, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

/// IBP bounds through PAN bottom-up path.
///
/// Tests progressive downsampling: P3 (8x8) -> merge with P4 (4x4)
/// -> downsample to P5 (2x2) + concat. Two stride-2 convolutions with
/// SiLU nonlinearity at each downsample stage.
#[test]
fn test_pan_bottom_up_ibp() {
    let def = build_pan_bottom_up_kernel();
    let bindings = pan_bottom_up_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[PAN3_P3_CHANNELS, PAN3_P3_SIZE, PAN3_P3_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN bottom-up");

    let out_channels = PAN3_P5_CHANNELS * 2;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[out_channels, PAN3_P5_SIZE, PAN3_P5_SIZE],
        "PAN bottom-up output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO PAN bottom-up IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 18. Multi-scale detection heads: detect at P3, P4, P5
// ===========================================================================

/// Smallest scale feature size for multi-scale detection.
const MS_P3_SIZE: usize = 8;
const MS_P3_CHANNELS: usize = 16;
/// Detection head output: cls sigmoid scores per anchor.
const MS_NUM_CLASSES: usize = 10;

/// Build a multi-scale detection cls head operating on 3 feature scales.
///
/// Tests a simplified version: 3 independent sigmoid classification heads
/// at different spatial scales (P3=8x8, P4=4x4, P5=2x2), all sharing
/// the same channel width. Outputs are flattened and concatenated.
///
/// Input (Variable): flattened P3+P4+P5 features.
/// Output: `[total_anchors, NUM_CLASSES]` class probabilities in [0, 1].
fn build_multiscale_cls_heads_kernel() -> TensorKernelDef {
    let c = MS_P3_CHANNELS;
    let s3 = MS_P3_SIZE; // 8
    let s4 = s3 / 2; // 4
    let s5 = s4 / 2; // 2

    let anchors3 = s3 * s3; // 64
    let anchors4 = s4 * s4; // 16
    let anchors5 = s5 * s5; // 4
    let total_anchors = anchors3 + anchors4 + anchors5; // 84

    let cls = MS_NUM_CLASSES;
    let mut b = TensorBlockBuilder::new("doclayout_multiscale_cls");

    // Input: concatenated features from 3 scales, flattened
    let input = b.add_input("multi_features", &[total_anchors, c]);

    // P3 head: narrow to [anchors3, c] -> conv-like linear -> sigmoid
    let p3_w = b.add_input("p3_cls_w", &[cls, c]);
    let p3_b = b.add_input("p3_cls_b", &[cls]);
    // Use matmul as linear projection: [anchors3+anchors4+anchors5, c] -> [total, cls]
    let logits = b.add_matmul(input, p3_w, true, None, &[total_anchors, cls]);
    // Add bias via broadcast
    let bias_bc = b.add_broadcast(p3_b, &[total_anchors, cls]);
    let biased = b.add_binary_add(logits, bias_bc, &[total_anchors, cls]);
    let out = b.add_sigmoid(biased, &[total_anchors, cls]);

    b.build(out)
        .expect("valid DocLayout-YOLO multi-scale cls heads kernel")
}

/// Bindings for multi-scale cls heads.
fn multiscale_cls_heads_bindings() -> Vec<TensorParamBinding> {
    let c = MS_P3_CHANNELS;
    let cls = MS_NUM_CLASSES;

    vec![
        TensorParamBinding::Variable, // multi_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[cls, c]), WEIGHT_MAG)), // p3_cls_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[cls]), 0.0f32)), // p3_cls_b
    ]
}

/// IBP bounds through multi-scale detection cls heads.
///
/// The final sigmoid ensures all class probabilities are in [0, 1] regardless
/// of the number of anchors from different feature scales.
#[test]
fn test_multiscale_cls_heads_ibp() {
    let s3 = MS_P3_SIZE;
    let s4 = s3 / 2;
    let s5 = s4 / 2;
    let total_anchors = s3 * s3 + s4 * s4 + s5 * s5;

    let def = build_multiscale_cls_heads_kernel();
    let bindings = multiscale_cls_heads_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[total_anchors, MS_P3_CHANNELS], 3.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-scale cls heads");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[total_anchors, MS_NUM_CLASSES],
        "multi-scale cls heads output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "DocLayout-YOLO multi-scale cls heads IBP ({total_anchors} anchors): bounds=[{lo_min}, {hi_max}]"
    );

    // Sigmoid codomain is (0, 1).
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper bound must be <= 1, got {hi_max}"
    );
}

/// CROWN bounds through multi-scale cls heads.
#[test]
fn test_multiscale_cls_heads_crown() {
    let s3 = MS_P3_SIZE;
    let s4 = s3 / 2;
    let s5 = s4 / 2;
    let total_anchors = s3 * s3 + s4 * s4 + s5 * s5;

    let def = build_multiscale_cls_heads_kernel();
    let bindings = multiscale_cls_heads_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[total_anchors, MS_P3_CHANNELS], 3.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[total_anchors, MS_NUM_CLASSES],
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "DocLayout-YOLO multi-scale cls CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "sigmoid lb >= 0 under CROWN");
    assert!(hi_max <= 1.0 + eps, "sigmoid ub <= 1 under CROWN");
}

// ===========================================================================
// 19. DFL + sigmoid compose: softmax decode then sigmoid confidence
// ===========================================================================

/// Build a DFL decode followed by sigmoid confidence score.
///
/// Input: `[NUM_ANCHORS, DFL_BINS]` (Variable, DFL logits).
/// Output: `[NUM_ANCHORS, 1]` (confidence score in [0, 1]).
///
/// Architecture:
///   Softmax(logits, dim=1) -> matmul(bins) -> sigmoid
///
/// The softmax converts logits to a distribution, matmul gives a continuous
/// coordinate, and the final sigmoid converts it to a confidence-like score.
/// Tests composition of softmax + linear + sigmoid nonlinearities.
fn build_dfl_sigmoid_compose_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("doclayout_dfl_sigmoid_compose");

    let input = b.add_input("dfl_logits", &[NUM_ANCHORS, DFL_BINS]);
    let bins = b.add_input("bins", &[DFL_BINS, 1]);

    // Softmax -> weighted sum
    let probs = b.add_softmax(input, 1, &[NUM_ANCHORS, DFL_BINS]);
    let coords = b.add_matmul(probs, bins, false, None, &[NUM_ANCHORS, 1]);

    // Sigmoid on the coordinate (models objectness confidence)
    let out = b.add_sigmoid(coords, &[NUM_ANCHORS, 1]);

    b.build(out).expect("valid DFL + sigmoid compose kernel")
}

/// Bindings for DFL + sigmoid compose.
fn dfl_sigmoid_compose_bindings() -> Vec<TensorParamBinding> {
    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins"),
        ),
    ]
}

/// IBP through DFL + sigmoid: softmax -> matmul -> sigmoid.
///
/// Key verification property: the final sigmoid guarantees output in [0, 1]
/// regardless of input logit range.
#[test]
fn test_dfl_sigmoid_compose_ibp() {
    let def = build_dfl_sigmoid_compose_kernel();
    let bindings = dfl_sigmoid_compose_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, DFL_BINS], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DFL + sigmoid compose");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_ANCHORS, 1],
        "DFL+sigmoid output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO DFL+sigmoid compose IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper bound must be <= 1, got {hi_max}"
    );
}

/// CROWN through DFL + sigmoid compose.
///
/// CROWN linearizes both the softmax and the final sigmoid. The matmul
/// with constant bins is a linear operation that does not require
/// linearization.
#[test]
fn test_dfl_sigmoid_compose_crown() {
    let def = build_dfl_sigmoid_compose_kernel();
    let bindings = dfl_sigmoid_compose_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, DFL_BINS], 5.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_ANCHORS, 1]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO DFL+sigmoid compose: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 20. NMS post-processing bounds: score thresholding via sigmoid
// ===========================================================================

/// Build an NMS-like post-processing block.
///
/// Input: `[NUM_ANCHORS, NUM_CLASSES + 4]` (Variable: cls logits + box coords).
/// Output: `[NUM_ANCHORS, NUM_CLASSES]` (filtered class probabilities in [0, 1]).
///
/// NMS post-processing in YOLO-family models begins with sigmoid on class
/// logits. This tests that the sigmoid guarantees [0, 1] output bounds
/// even with wide input ranges that include box coordinate columns.
///
/// We narrow the input to the classification columns and apply sigmoid.
/// (Actual NMS thresholding and IoU suppression happen at inference time
/// and are outside the differentiable graph.)
fn build_nms_postprocess_kernel() -> TensorKernelDef {
    let n = NUM_ANCHORS;
    let cls = NUM_CLASSES;
    let total_cols = cls + 4; // cls logits + 4 box coords
    let mut b = TensorBlockBuilder::new("doclayout_nms_postprocess");

    let input = b.add_input("raw_detections", &[n, total_cols]);

    // Narrow to class columns: [NUM_ANCHORS, NUM_CLASSES+4] -> [NUM_ANCHORS, NUM_CLASSES]
    // Use reshape to model the narrow (take first NUM_CLASSES columns)
    // Since TensorBlockBuilder doesn't have a native narrow on dim=1 for 2D,
    // we model this as a linear projection that selects class columns.
    let select_w = b.add_input("select_w", &[cls, total_cols]);
    let cls_logits = b.add_matmul(input, select_w, true, None, &[n, cls]);

    // Sigmoid: class probabilities in [0, 1]
    let out = b.add_sigmoid(cls_logits, &[n, cls]);

    b.build(out).expect("valid NMS post-processing kernel")
}

/// Bindings for NMS post-processing.
///
/// The select_w is an identity-like matrix that selects the first
/// NUM_CLASSES columns from the input.
fn nms_postprocess_bindings() -> Vec<TensorParamBinding> {
    let cls = NUM_CLASSES;
    let total_cols = cls + 4;

    // Identity-like selector: each row i has 1.0 at column i, 0.0 elsewhere
    let mut select_data = vec![0.0f32; cls * total_cols];
    for i in 0..cls {
        select_data[i * total_cols + i] = 1.0;
    }
    let select_w =
        ArrayD::from_shape_vec(IxDyn(&[cls, total_cols]), select_data).expect("valid select shape");

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(select_w),
    ]
}

/// IBP bounds through NMS post-processing.
///
/// Even with wide input bounds (raw detections including box coordinates
/// in arbitrary range), the sigmoid guarantees output class probabilities
/// are in [0, 1].
#[test]
fn test_nms_postprocess_ibp() {
    let def = build_nms_postprocess_kernel();
    let bindings = nms_postprocess_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, NUM_CLASSES + 4], 20.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through NMS post-processing");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_ANCHORS, NUM_CLASSES],
        "NMS postprocess output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "DocLayout-YOLO NMS postprocess IBP (wide input [-20,20]): bounds=[{lo_min}, {hi_max}]"
    );

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "sigmoid lb >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "sigmoid ub <= 1, got {hi_max}");
}

// ===========================================================================
// 21. Backbone + SPPF + C2f compose
// ===========================================================================

/// Build a backbone stage -> SPPF -> C2f simplified compose.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image [0, 1]).
/// Output: `[BOTTLENECK_CHANNELS, SPPF_SIZE, SPPF_SIZE]`.
///
/// Architecture:
///   ConvBnAct(3, 16, 3, s=2, p=1) -> [16, 16, 16]
///   ConvBnAct(16, 16, 3, s=2, p=1) -> [16, 8, 8]
///   SPPF: 3x MaxPool + concat -> [64, 8, 8]
///   1x1 conv reduction -> [16, 8, 8]
///   Bottleneck residual: Conv3x3 -> BN -> SiLU + skip -> [16, 8, 8]
fn build_backbone_sppf_c2f_kernel() -> TensorKernelDef {
    let c0 = CONV_OUT_CHANNELS; // 16
    let c_bn = BOTTLENECK_CHANNELS; // 16
    let s0 = CONV_OUT_SIZE; // 16
    let s1 = BOTTLENECK_SIZE; // 8
    let sppf_out_c = c_bn * 4; // 64

    let mut b = TensorBlockBuilder::new("doclayout_backbone_sppf_c2f");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Stage 0: ConvBnAct(3, 16, 3, s=2, p=1) -> [16, 16, 16]
    let s0_w = b.add_input("s0_w", &[c0, IN_CHANNELS, 3, 3]);
    let s0_b = b.add_input("s0_b", &[c0]);
    let s0_bn_m = b.add_input("s0_bn_m", &[c0]);
    let s0_bn_v = b.add_input("s0_bn_v", &[c0]);
    let s0_bn_w = b.add_input("s0_bn_w", &[c0]);
    let s0_bn_b = b.add_input("s0_bn_b", &[c0]);
    let s0_bn_e = b.add_input("s0_bn_e", &[1]);

    let conv0 = b.add_conv2d(input, s0_w, Some(s0_b), 2, 2, 1, 1, &[c0, s0, s0]);
    let bn0 = b.add_batch_norm(
        conv0,
        s0_bn_m,
        s0_bn_v,
        s0_bn_w,
        s0_bn_b,
        s0_bn_e,
        &[c0, s0, s0],
    );
    let sig0 = b.add_sigmoid(bn0, &[c0, s0, s0]);
    let silu0 = b.add_binary_mul(bn0, sig0, &[c0, s0, s0]);

    // Stage 1: ConvBnAct(16, 16, 3, s=2, p=1) -> [16, 8, 8]
    let s1_w = b.add_input("s1_w", &[c_bn, c0, 3, 3]);
    let s1_b = b.add_input("s1_b", &[c_bn]);
    let s1_bn_m = b.add_input("s1_bn_m", &[c_bn]);
    let s1_bn_v = b.add_input("s1_bn_v", &[c_bn]);
    let s1_bn_w = b.add_input("s1_bn_w", &[c_bn]);
    let s1_bn_b = b.add_input("s1_bn_b", &[c_bn]);
    let s1_bn_e = b.add_input("s1_bn_e", &[1]);

    let conv1 = b.add_conv2d(silu0, s1_w, Some(s1_b), 2, 2, 1, 1, &[c_bn, s1, s1]);
    let bn1 = b.add_batch_norm(
        conv1,
        s1_bn_m,
        s1_bn_v,
        s1_bn_w,
        s1_bn_b,
        s1_bn_e,
        &[c_bn, s1, s1],
    );
    let sig1 = b.add_sigmoid(bn1, &[c_bn, s1, s1]);
    let silu1 = b.add_binary_mul(bn1, sig1, &[c_bn, s1, s1]);

    // SPPF: 3x MaxPool(k=5, s=1, p=2) + concat
    let pool1 = b.add_max_pool_2d(
        silu1,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &[c_bn, s1, s1],
    );
    let pool2 = b.add_max_pool_2d(
        pool1,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &[c_bn, s1, s1],
    );
    let pool3 = b.add_max_pool_2d(
        pool2,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &[c_bn, s1, s1],
    );
    let sppf = b.add_concat(&[silu1, pool1, pool2, pool3], 0, &[sppf_out_c, s1, s1]);

    // 1x1 conv reduction: [64, 8, 8] -> [16, 8, 8]
    let red_w = b.add_input("red_w", &[c_bn, sppf_out_c, 1, 1]);
    let red_b = b.add_input("red_b", &[c_bn]);
    let reduced = b.add_conv2d(sppf, red_w, Some(red_b), 1, 1, 0, 0, &[c_bn, s1, s1]);

    // Bottleneck residual: Conv3x3 -> BN -> SiLU + skip
    let bn_w = b.add_input("bn_conv_w", &[c_bn, c_bn, 3, 3]);
    let bn_cb = b.add_input("bn_conv_b", &[c_bn]);
    let bn_bm = b.add_input("bn_bm", &[c_bn]);
    let bn_bv = b.add_input("bn_bv", &[c_bn]);
    let bn_bw = b.add_input("bn_bw", &[c_bn]);
    let bn_bb = b.add_input("bn_bb", &[c_bn]);
    let bn_be = b.add_input("bn_be", &[1]);

    let bn_conv = b.add_conv2d(reduced, bn_w, Some(bn_cb), 1, 1, 1, 1, &[c_bn, s1, s1]);
    let bn_bn = b.add_batch_norm(bn_conv, bn_bm, bn_bv, bn_bw, bn_bb, bn_be, &[c_bn, s1, s1]);
    let bn_sig = b.add_sigmoid(bn_bn, &[c_bn, s1, s1]);
    let bn_silu = b.add_binary_mul(bn_bn, bn_sig, &[c_bn, s1, s1]);
    let out = b.add_binary_add(bn_silu, reduced, &[c_bn, s1, s1]);

    b.build(out).expect("valid backbone + SPPF + C2f kernel")
}

/// Bindings for backbone + SPPF + C2f.
fn backbone_sppf_c2f_bindings() -> Vec<TensorParamBinding> {
    let c0 = CONV_OUT_CHANNELS;
    let c_bn = BOTTLENECK_CHANNELS;
    let sppf_out_c = c_bn * 4;

    vec![
        TensorParamBinding::Variable, // image
        // Stage 0
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c0, IN_CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // Stage 1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn, c0, 3, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // Reduction conv
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c_bn, sppf_out_c, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 0.0f32)),
        // Bottleneck conv + BN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c_bn, c_bn, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

/// IBP bounds through backbone + SPPF + C2f compose.
///
/// The deepest sequential composition: image -> 2 ConvBnAct stages ->
/// SPPF multi-scale pooling -> 1x1 reduction -> bottleneck residual.
/// Tests that IBP propagates cleanly through conv+BN+SiLU, MaxPool chain,
/// concat, channel reduction, and residual skip connection.
#[test]
fn test_backbone_sppf_c2f_ibp() {
    let def = build_backbone_sppf_c2f_kernel();
    let bindings = backbone_sppf_c2f_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through backbone + SPPF + C2f");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        "backbone+SPPF+C2f output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO backbone+SPPF+C2f IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 22. Full detection pipeline: backbone -> SPPF -> detect head (cls+box)
// ===========================================================================

/// Build the complete detection pipeline with backbone, SPPF, and both heads.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image [0, 1]).
/// Output: `[FULL_ANCHORS, NUM_CLASSES + 1]` (cls probs + 1 box DFL coord).
///
/// Architecture:
///   ConvBnAct(3, 16, 3, s=2, p=1) -> ConvBnAct(16, 16, 3, s=2, p=1) -> [16, 8, 8]
///   SPPF -> [64, 8, 8] -> 1x1 reduce -> [16, 8, 8]
///   Cls branch: 1x1 conv -> reshape -> sigmoid -> [64, 10]
///   Box branch: 1x1 conv -> reshape -> softmax -> DFL -> [64, 1]
///   Concat -> [64, 11]
const FULL_ANCHORS: usize = BOTTLENECK_SIZE * BOTTLENECK_SIZE; // 64

fn build_full_detection_pipeline_kernel() -> TensorKernelDef {
    let c0 = CONV_OUT_CHANNELS; // 16
    let c_bn = BOTTLENECK_CHANNELS; // 16
    let s0 = CONV_OUT_SIZE; // 16
    let s1 = BOTTLENECK_SIZE; // 8
    let sppf_out_c = c_bn * 4; // 64
    let cls = NUM_CLASSES;
    let n_anchors = s1 * s1; // 64

    let mut b = TensorBlockBuilder::new("doclayout_full_detection");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Stage 0: ConvBnAct(3, 16, 3, s=2, p=1) -> SiLU
    let s0_w = b.add_input("s0_w", &[c0, IN_CHANNELS, 3, 3]);
    let s0_b = b.add_input("s0_b", &[c0]);
    let s0_bn_m = b.add_input("s0_bn_m", &[c0]);
    let s0_bn_v = b.add_input("s0_bn_v", &[c0]);
    let s0_bn_w = b.add_input("s0_bn_w", &[c0]);
    let s0_bn_b = b.add_input("s0_bn_b", &[c0]);
    let s0_bn_e = b.add_input("s0_bn_e", &[1]);

    let conv0 = b.add_conv2d(input, s0_w, Some(s0_b), 2, 2, 1, 1, &[c0, s0, s0]);
    let bn0 = b.add_batch_norm(
        conv0,
        s0_bn_m,
        s0_bn_v,
        s0_bn_w,
        s0_bn_b,
        s0_bn_e,
        &[c0, s0, s0],
    );
    let sig0 = b.add_sigmoid(bn0, &[c0, s0, s0]);
    let silu0 = b.add_binary_mul(bn0, sig0, &[c0, s0, s0]);

    // Stage 1: ConvBnAct(16, 16, 3, s=2, p=1) -> SiLU
    let s1_w = b.add_input("s1_w", &[c_bn, c0, 3, 3]);
    let s1_b = b.add_input("s1_b", &[c_bn]);
    let s1_bn_m = b.add_input("s1_bn_m", &[c_bn]);
    let s1_bn_v = b.add_input("s1_bn_v", &[c_bn]);
    let s1_bn_w = b.add_input("s1_bn_w", &[c_bn]);
    let s1_bn_b = b.add_input("s1_bn_b", &[c_bn]);
    let s1_bn_e = b.add_input("s1_bn_e", &[1]);

    let conv1 = b.add_conv2d(silu0, s1_w, Some(s1_b), 2, 2, 1, 1, &[c_bn, s1, s1]);
    let bn1 = b.add_batch_norm(
        conv1,
        s1_bn_m,
        s1_bn_v,
        s1_bn_w,
        s1_bn_b,
        s1_bn_e,
        &[c_bn, s1, s1],
    );
    let sig1 = b.add_sigmoid(bn1, &[c_bn, s1, s1]);
    let silu1 = b.add_binary_mul(bn1, sig1, &[c_bn, s1, s1]);

    // SPPF
    let pool1 = b.add_max_pool_2d(
        silu1,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &[c_bn, s1, s1],
    );
    let pool2 = b.add_max_pool_2d(
        pool1,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &[c_bn, s1, s1],
    );
    let pool3 = b.add_max_pool_2d(
        pool2,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &[c_bn, s1, s1],
    );
    let sppf = b.add_concat(&[silu1, pool1, pool2, pool3], 0, &[sppf_out_c, s1, s1]);

    // 1x1 reduce: [64, 8, 8] -> [16, 8, 8]
    let red_w = b.add_input("red_w", &[c_bn, sppf_out_c, 1, 1]);
    let red_b = b.add_input("red_b", &[c_bn]);
    let reduced = b.add_conv2d(sppf, red_w, Some(red_b), 1, 1, 0, 0, &[c_bn, s1, s1]);

    // Cls head: 1x1 conv -> reshape -> sigmoid
    let cls_w = b.add_input("cls_w", &[cls, c_bn, 1, 1]);
    let cls_b = b.add_input("cls_b", &[cls]);
    let cls_conv = b.add_conv2d(reduced, cls_w, Some(cls_b), 1, 1, 0, 0, &[cls, s1, s1]);
    let cls_flat = b.add_reshape(cls_conv, &[n_anchors, cls]);
    let cls_probs = b.add_sigmoid(cls_flat, &[n_anchors, cls]);

    // Box head: 1x1 conv -> reshape -> softmax -> DFL decode
    let box_w = b.add_input("box_w", &[DFL_BINS, c_bn, 1, 1]);
    let box_b = b.add_input("box_b", &[DFL_BINS]);
    let bins_input = b.add_input("dfl_bins", &[DFL_BINS, 1]);

    let box_conv = b.add_conv2d(reduced, box_w, Some(box_b), 1, 1, 0, 0, &[DFL_BINS, s1, s1]);
    let box_flat = b.add_reshape(box_conv, &[n_anchors, DFL_BINS]);
    let box_probs = b.add_softmax(box_flat, 1, &[n_anchors, DFL_BINS]);
    let box_coords = b.add_matmul(box_probs, bins_input, false, None, &[n_anchors, 1]);

    // Concat cls + box
    let out = b.add_concat(&[cls_probs, box_coords], 1, &[n_anchors, cls + 1]);

    b.build(out).expect("valid full detection pipeline kernel")
}

/// Bindings for full detection pipeline.
fn full_detection_pipeline_bindings() -> Vec<TensorParamBinding> {
    let c0 = CONV_OUT_CHANNELS;
    let c_bn = BOTTLENECK_CHANNELS;
    let sppf_out_c = c_bn * 4;
    let cls = NUM_CLASSES;
    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();

    vec![
        TensorParamBinding::Variable, // image
        // Stage 0
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c0, IN_CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c0]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // Stage 1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn, c0, 3, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // Reduce conv
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c_bn, sppf_out_c, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_bn]), 0.0f32)),
        // Cls head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[cls, c_bn, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[cls]), 0.0f32)),
        // Box head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DFL_BINS, c_bn, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DFL_BINS]), 0.0f32)),
        // DFL bins
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins"),
        ),
    ]
}

/// IBP bounds through full detection pipeline.
///
/// Most comprehensive compose test: image -> backbone (2 ConvBnAct stages)
/// -> SPPF (3x MaxPool + concat) -> reduce -> parallel cls sigmoid +
/// box DFL decode -> concat output.
#[test]
fn test_full_detection_pipeline_ibp() {
    let def = build_full_detection_pipeline_kernel();
    let bindings = full_detection_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full detection pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[FULL_ANCHORS, NUM_CLASSES + 1],
        "full detection pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "DocLayout-YOLO full detection pipeline IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 23. Verify-and-record: C2f 3-bottleneck
// ===========================================================================

/// Verify and record C2f 3-bottleneck chain.
#[test]
fn test_c2f_three_bottleneck_verify_and_record() {
    let def = build_c2f_three_bottleneck_kernel();
    let bindings = c2f_three_bottleneck_bindings();
    let input = uniform_bounds(
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        2.0,
    );

    let result = verify_and_assert(&def, &bindings, &input, "doclayout_c2f_3bn");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(
        lo.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
    );
}

// ===========================================================================
// 24. Verify-and-record: backbone + SPPF + C2f
// ===========================================================================

/// Verify and record backbone + SPPF + C2f end-to-end.
#[test]
fn test_backbone_sppf_c2f_verify_and_record() {
    let def = build_backbone_sppf_c2f_kernel();
    let bindings = backbone_sppf_c2f_bindings();
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(&def, &bindings, &input, "doclayout_backbone_sppf_c2f");
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(
        lo.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
    );
}

// ===========================================================================
// 25. Verify-and-record: full detection pipeline
// ===========================================================================

/// Verify and record full detection pipeline.
#[test]
fn test_full_detection_pipeline_verify_and_record() {
    let def = build_full_detection_pipeline_kernel();
    let bindings = full_detection_pipeline_bindings();
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(&def, &bindings, &input, "doclayout_full_detection_pipeline");
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[FULL_ANCHORS, NUM_CLASSES + 1]);
}

// ===========================================================================
// ===========================================================================
//
//  DEEP COMPOSE TESTS (Part of #3952)
//
//  26–43: Deeper backbone, PAN neck, and detection head compositions.
//
// ===========================================================================
// ===========================================================================

// ---------------------------------------------------------------------------
// Dimensions for deep tests
// ---------------------------------------------------------------------------

/// Backbone stage channels for 4-stage cascade with C2f.
const DEEP_BB_C: [usize; 4] = [8, 16, 32, 64];
/// Spatial sizes for 4-stage cascade: 16 -> 8 -> 4 -> 2.
const DEEP_BB_S: [usize; 4] = [16, 8, 4, 2];
/// Feature pyramid output channels for PAN (P3/P4/P5).
const FPN_P3_C: usize = 16;
const FPN_P4_C: usize = 32;
const FPN_P5_C: usize = 64;
const FPN_P3_S: usize = 8;
const FPN_P4_S: usize = 4;
const FPN_P5_S: usize = 2;
/// Anchor-free detection head channels.
const AF_HEAD_HIDDEN: usize = 32;
/// Deep C2f bottleneck count.
const DEEP_C2F_BOTTLENECKS: usize = 3;

// ===========================================================================
// 26. C2f with 3 bottleneck splits and channel-split concat (full fidelity)
// ===========================================================================

/// Build a C2f block with explicit channel split: entry conv halves channels,
/// the split half passes through 3 bottlenecks, then all branches concat.
///
/// Input: `[32, 4, 4]` (Variable).
/// Output: `[32, 4, 4]`.
///
/// Architecture:
///   Entry 1x1 conv -> BN -> SiLU (C=32 -> C=32)
///   Split conceptually: pass-through branch + bottleneck branch
///   Bottleneck 1: 3x3 conv -> BN -> SiLU + skip
///   Bottleneck 2: 3x3 conv -> BN -> SiLU + skip
///   Bottleneck 3: 3x3 conv -> BN -> SiLU + skip
///   Concat(pass-through, bn1, bn2, bn3) -> C*4
///   Exit 1x1 conv (C*4 -> C) -> BN -> SiLU
fn build_c2f_split_three_bottleneck_kernel() -> TensorKernelDef {
    let c = DEEP_BB_C[1]; // 16 — use half of 32 for tractability
    let s = DEEP_BB_S[1]; // 8
    let feat_shape = [c, s, s];
    let mut b = TensorBlockBuilder::new("deep_c2f_split_3bn");

    let input = b.add_input("features", &feat_shape);

    // Entry 1x1 conv -> BN -> SiLU
    let ew = b.add_input("entry_w", &[c, c, 1, 1]);
    let eb = b.add_input("entry_b", &[c]);
    let ebm = b.add_input("entry_bm", &[c]);
    let ebv = b.add_input("entry_bv", &[c]);
    let ebw = b.add_input("entry_bw", &[c]);
    let ebb = b.add_input("entry_bb", &[c]);
    let ebe = b.add_input("entry_be", &[1]);

    let ec = b.add_conv2d(input, ew, Some(eb), 1, 1, 0, 0, &feat_shape);
    let ebn = b.add_batch_norm(ec, ebm, ebv, ebw, ebb, ebe, &feat_shape);
    let esig = b.add_sigmoid(ebn, &feat_shape);
    let esilu = b.add_binary_mul(ebn, esig, &feat_shape);

    // 3 bottleneck residual blocks
    let mut prev = esilu;
    let mut branches = vec![esilu]; // pass-through branch

    for i in 0..DEEP_C2F_BOTTLENECKS {
        let prefix = format!("bn{i}");
        let cw = b.add_input(&format!("{prefix}_cw"), &[c, c, 3, 3]);
        let cb = b.add_input(&format!("{prefix}_cb"), &[c]);
        let bm = b.add_input(&format!("{prefix}_bm"), &[c]);
        let bv = b.add_input(&format!("{prefix}_bv"), &[c]);
        let bw = b.add_input(&format!("{prefix}_bw"), &[c]);
        let bb = b.add_input(&format!("{prefix}_bb"), &[c]);
        let be = b.add_input(&format!("{prefix}_be"), &[1]);

        let conv = b.add_conv2d(prev, cw, Some(cb), 1, 1, 1, 1, &feat_shape);
        let bn = b.add_batch_norm(conv, bm, bv, bw, bb, be, &feat_shape);
        let sig = b.add_sigmoid(bn, &feat_shape);
        let silu = b.add_binary_mul(bn, sig, &feat_shape);
        let res = b.add_binary_add(silu, prev, &feat_shape);

        branches.push(res);
        prev = res;
    }

    // Concat all branches: pass-through + 3 bottleneck outputs = 4*C channels
    let concat_c = c * 4;
    let concat_shape = [concat_c, s, s];
    let cat = b.add_concat(&branches, 0, &concat_shape);

    // Exit 1x1 conv: C*4 -> C -> BN -> SiLU
    let xw = b.add_input("exit_w", &[c, concat_c, 1, 1]);
    let xb = b.add_input("exit_b", &[c]);
    let xbm = b.add_input("exit_bm", &[c]);
    let xbv = b.add_input("exit_bv", &[c]);
    let xbw = b.add_input("exit_bw", &[c]);
    let xbb = b.add_input("exit_bb", &[c]);
    let xbe = b.add_input("exit_be", &[1]);

    let xc = b.add_conv2d(cat, xw, Some(xb), 1, 1, 0, 0, &feat_shape);
    let xbn = b.add_batch_norm(xc, xbm, xbv, xbw, xbb, xbe, &feat_shape);
    let xsig = b.add_sigmoid(xbn, &feat_shape);
    let out = b.add_binary_mul(xbn, xsig, &feat_shape);

    b.build(out)
        .expect("valid deep C2f split 3-bottleneck kernel")
}

fn c2f_split_three_bottleneck_bindings() -> Vec<TensorParamBinding> {
    let c = DEEP_BB_C[1];
    let concat_c = c * 4;
    let c1x1 = |ci: usize, co: usize| ArrayD::from_elem(IxDyn(&[co, ci, 1, 1]), WEIGHT_MAG);
    let c3x3 = || ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG);
    let z = || ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let o = || ArrayD::from_elem(IxDyn(&[c]), 1.0f32);

    let mut v = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(c1x1(c, c)),
        TensorParamBinding::ConstantTensor(z()),
        TensorParamBinding::ConstantTensor(z()),
        TensorParamBinding::ConstantTensor(o()),
        TensorParamBinding::ConstantTensor(o()),
        TensorParamBinding::ConstantTensor(z()),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    for _ in 0..DEEP_C2F_BOTTLENECKS {
        v.push(TensorParamBinding::ConstantTensor(c3x3()));
        v.push(TensorParamBinding::ConstantTensor(z()));
        v.push(TensorParamBinding::ConstantTensor(z()));
        v.push(TensorParamBinding::ConstantTensor(o()));
        v.push(TensorParamBinding::ConstantTensor(o()));
        v.push(TensorParamBinding::ConstantTensor(z()));
        v.push(TensorParamBinding::ConstantScalar(1e-5));
    }
    v.push(TensorParamBinding::ConstantTensor(c1x1(concat_c, c)));
    v.push(TensorParamBinding::ConstantTensor(z()));
    v.push(TensorParamBinding::ConstantTensor(z()));
    v.push(TensorParamBinding::ConstantTensor(o()));
    v.push(TensorParamBinding::ConstantTensor(o()));
    v.push(TensorParamBinding::ConstantTensor(z()));
    v.push(TensorParamBinding::ConstantScalar(1e-5));
    v
}

/// IBP through C2f with explicit 4-way channel split+concat and 3 bottlenecks.
#[test]
fn test_c2f_split_three_bottleneck_ibp() {
    let def = build_c2f_split_three_bottleneck_kernel();
    let bindings = c2f_split_three_bottleneck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let c = DEEP_BB_C[1];
    let s = DEEP_BB_S[1];
    let input = uniform_bounds(&[c, s, s], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through C2f split 3-bottleneck");

    assert_eq!(output.lower_upper().0.shape(), &[c, s, s]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Deep C2f split 3-bn IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite());
    assert!(hi_max.is_finite());
}

/// CROWN through C2f split 3-bottleneck: linearizes 3*SiLU + entry SiLU + exit SiLU.
#[test]
fn test_c2f_split_three_bottleneck_crown() {
    let def = build_c2f_split_three_bottleneck_kernel();
    let bindings = c2f_split_three_bottleneck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let c = DEEP_BB_C[1];
    let s = DEEP_BB_S[1];
    let input = uniform_bounds(&[c, s, s], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[c, s, s]);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Deep C2f split 3-bn CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 28. 4-stage ConvBnAct + C2f cascaded downsampling
// ===========================================================================

/// Build 4-stage backbone with ConvBnAct downsampling at each stage, plus a
/// C2f block (1 bottleneck) at the final stage.
///
/// Input: `[3, 32, 32]` image.
/// Output: `[64, 2, 2]` after 4 stages + C2f.
///
/// Architecture:
///   Stage 0: Conv(3, 8, 3, s=2, p=1) -> BN -> SiLU -> [8, 16, 16]
///   Stage 1: Conv(8, 16, 3, s=2, p=1) -> BN -> SiLU -> [16, 8, 8]
///   Stage 2: Conv(16, 32, 3, s=2, p=1) -> BN -> SiLU -> [32, 4, 4]
///   Stage 3: Conv(32, 64, 3, s=2, p=1) -> BN -> SiLU -> [64, 2, 2]
///   C2f: 1x1 -> bottleneck -> concat -> 1x1 -> BN -> SiLU -> [64, 2, 2]
fn build_four_stage_c2f_cascade_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("deep_4stage_c2f_cascade");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let mut prev = input;
    let mut prev_c = IN_CHANNELS;

    // 4 ConvBnAct stages
    for stage in 0..4 {
        let co = DEEP_BB_C[stage];
        let so = DEEP_BB_S[stage];
        let shape = [co, so, so];
        let p = format!("s{stage}");

        let w = b.add_input(&format!("{p}_w"), &[co, prev_c, 3, 3]);
        let cb = b.add_input(&format!("{p}_b"), &[co]);
        let bm = b.add_input(&format!("{p}_bm"), &[co]);
        let bv = b.add_input(&format!("{p}_bv"), &[co]);
        let bw = b.add_input(&format!("{p}_bw"), &[co]);
        let bb = b.add_input(&format!("{p}_bb"), &[co]);
        let be = b.add_input(&format!("{p}_be"), &[1]);

        let conv = b.add_conv2d(prev, w, Some(cb), 2, 2, 1, 1, &shape);
        let bn = b.add_batch_norm(conv, bm, bv, bw, bb, be, &shape);
        let sig = b.add_sigmoid(bn, &shape);
        let silu = b.add_binary_mul(bn, sig, &shape);

        prev = silu;
        prev_c = co;
    }

    // C2f at final stage: entry 1x1 -> 1 bottleneck -> concat -> exit 1x1
    let c = DEEP_BB_C[3]; // 64
    let s = DEEP_BB_S[3]; // 2
    let feat = [c, s, s];

    let ew = b.add_input("c2f_entry_w", &[c, c, 1, 1]);
    let eb = b.add_input("c2f_entry_b", &[c]);
    let ebm = b.add_input("c2f_entry_bm", &[c]);
    let ebv = b.add_input("c2f_entry_bv", &[c]);
    let ebw = b.add_input("c2f_entry_bw", &[c]);
    let ebb = b.add_input("c2f_entry_bb", &[c]);
    let ebe = b.add_input("c2f_entry_be", &[1]);

    let ec = b.add_conv2d(prev, ew, Some(eb), 1, 1, 0, 0, &feat);
    let ebn = b.add_batch_norm(ec, ebm, ebv, ebw, ebb, ebe, &feat);
    let esig = b.add_sigmoid(ebn, &feat);
    let esilu = b.add_binary_mul(ebn, esig, &feat);

    // Single bottleneck
    let bncw = b.add_input("c2f_bn_cw", &[c, c, 3, 3]);
    let bncb = b.add_input("c2f_bn_cb", &[c]);
    let bnbm = b.add_input("c2f_bn_bm", &[c]);
    let bnbv = b.add_input("c2f_bn_bv", &[c]);
    let bnbw = b.add_input("c2f_bn_bw", &[c]);
    let bnbb = b.add_input("c2f_bn_bb", &[c]);
    let bnbe = b.add_input("c2f_bn_be", &[1]);

    let bnconv = b.add_conv2d(esilu, bncw, Some(bncb), 1, 1, 1, 1, &feat);
    let bnbn = b.add_batch_norm(bnconv, bnbm, bnbv, bnbw, bnbb, bnbe, &feat);
    let bnsig = b.add_sigmoid(bnbn, &feat);
    let bnsilu = b.add_binary_mul(bnbn, bnsig, &feat);
    let bnres = b.add_binary_add(bnsilu, esilu, &feat);

    // Concat entry + bottleneck -> 2C
    let cat_c = c * 2;
    let cat_shape = [cat_c, s, s];
    let cat = b.add_concat(&[esilu, bnres], 0, &cat_shape);

    // Exit 1x1: 2C -> C -> BN -> SiLU
    let xw = b.add_input("c2f_exit_w", &[c, cat_c, 1, 1]);
    let xb = b.add_input("c2f_exit_b", &[c]);
    let xbm = b.add_input("c2f_exit_bm", &[c]);
    let xbv = b.add_input("c2f_exit_bv", &[c]);
    let xbw = b.add_input("c2f_exit_bw", &[c]);
    let xbb = b.add_input("c2f_exit_bb", &[c]);
    let xbe = b.add_input("c2f_exit_be", &[1]);

    let xc = b.add_conv2d(cat, xw, Some(xb), 1, 1, 0, 0, &feat);
    let xbn = b.add_batch_norm(xc, xbm, xbv, xbw, xbb, xbe, &feat);
    let xsig = b.add_sigmoid(xbn, &feat);
    let out = b.add_binary_mul(xbn, xsig, &feat);

    b.build(out).expect("valid 4-stage+C2f cascade kernel")
}

fn four_stage_c2f_cascade_bindings() -> Vec<TensorParamBinding> {
    let mut v = vec![TensorParamBinding::Variable]; // image
    let mut prev_c = IN_CHANNELS;

    // 4 ConvBnAct stages
    for &co in &DEEP_BB_C {
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[co, prev_c, 3, 3]),
            WEIGHT_MAG,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[co]),
            0.0f32,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[co]),
            0.0f32,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[co]),
            1.0f32,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[co]),
            1.0f32,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[co]),
            0.0f32,
        )));
        v.push(TensorParamBinding::ConstantScalar(1e-5));
        prev_c = co;
    }

    let c = DEEP_BB_C[3]; // 64
    let cat_c = c * 2;

    // C2f entry 1x1
    let z = || ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let o = || ArrayD::from_elem(IxDyn(&[c]), 1.0f32);

    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c, c, 1, 1]),
        WEIGHT_MAG,
    )));
    v.push(TensorParamBinding::ConstantTensor(z()));
    v.push(TensorParamBinding::ConstantTensor(z()));
    v.push(TensorParamBinding::ConstantTensor(o()));
    v.push(TensorParamBinding::ConstantTensor(o()));
    v.push(TensorParamBinding::ConstantTensor(z()));
    v.push(TensorParamBinding::ConstantScalar(1e-5));

    // Bottleneck 3x3
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c, c, 3, 3]),
        WEIGHT_MAG,
    )));
    v.push(TensorParamBinding::ConstantTensor(z()));
    v.push(TensorParamBinding::ConstantTensor(z()));
    v.push(TensorParamBinding::ConstantTensor(o()));
    v.push(TensorParamBinding::ConstantTensor(o()));
    v.push(TensorParamBinding::ConstantTensor(z()));
    v.push(TensorParamBinding::ConstantScalar(1e-5));

    // Exit 1x1
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c, cat_c, 1, 1]),
        WEIGHT_MAG,
    )));
    v.push(TensorParamBinding::ConstantTensor(z()));
    v.push(TensorParamBinding::ConstantTensor(z()));
    v.push(TensorParamBinding::ConstantTensor(o()));
    v.push(TensorParamBinding::ConstantTensor(o()));
    v.push(TensorParamBinding::ConstantTensor(z()));
    v.push(TensorParamBinding::ConstantScalar(1e-5));

    v
}

/// IBP through 4-stage backbone with C2f at final stage.
///
/// Tests deep sequential propagation: 4 ConvBnAct downsampling stages
/// followed by a C2f block with entry conv, bottleneck residual, concat,
/// and exit conv. Total of 6 SiLU nonlinearities.
#[test]
fn test_four_stage_c2f_cascade_ibp() {
    let def = build_four_stage_c2f_cascade_kernel();
    let bindings = four_stage_c2f_cascade_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-stage+C2f cascade");

    let c = DEEP_BB_C[3];
    let s = DEEP_BB_S[3];
    assert_eq!(output.lower_upper().0.shape(), &[c, s, s]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Deep 4-stage+C2f cascade IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite());
    assert!(hi_max.is_finite());
}

// ===========================================================================
// 29. SPPF at full depth: backbone -> SPPF -> 1x1 reduce -> BN -> SiLU
// ===========================================================================

/// Build backbone 2-stage -> SPPF -> reduction conv -> BN -> SiLU.
///
/// Input: `[3, 32, 32]` image.
/// Output: `[16, 8, 8]` reduced features.
///
/// Tests SPPF MaxPool chain propagation at full backbone depth.
fn build_deep_sppf_reduce_kernel() -> TensorKernelDef {
    let c0 = CONV_OUT_CHANNELS; // 16
    let c1 = BOTTLENECK_CHANNELS; // 16
    let s0 = CONV_OUT_SIZE; // 16
    let s1 = BOTTLENECK_SIZE; // 8
    let sppf_c = c1 * 4; // 64

    let mut b = TensorBlockBuilder::new("deep_sppf_reduce");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Stage 0: Conv(3, 16, 3, s=2, p=1) -> BN -> SiLU
    let s0w = b.add_input("s0_w", &[c0, IN_CHANNELS, 3, 3]);
    let s0b = b.add_input("s0_b", &[c0]);
    let s0bm = b.add_input("s0_bm", &[c0]);
    let s0bv = b.add_input("s0_bv", &[c0]);
    let s0bw = b.add_input("s0_bw", &[c0]);
    let s0bb = b.add_input("s0_bb", &[c0]);
    let s0be = b.add_input("s0_be", &[1]);

    let conv0 = b.add_conv2d(input, s0w, Some(s0b), 2, 2, 1, 1, &[c0, s0, s0]);
    let bn0 = b.add_batch_norm(conv0, s0bm, s0bv, s0bw, s0bb, s0be, &[c0, s0, s0]);
    let sig0 = b.add_sigmoid(bn0, &[c0, s0, s0]);
    let silu0 = b.add_binary_mul(bn0, sig0, &[c0, s0, s0]);

    // Stage 1: Conv(16, 16, 3, s=2, p=1) -> BN -> SiLU
    let s1w = b.add_input("s1_w", &[c1, c0, 3, 3]);
    let s1b = b.add_input("s1_b", &[c1]);
    let s1bm = b.add_input("s1_bm", &[c1]);
    let s1bv = b.add_input("s1_bv", &[c1]);
    let s1bw = b.add_input("s1_bw", &[c1]);
    let s1bb = b.add_input("s1_bb", &[c1]);
    let s1be = b.add_input("s1_be", &[1]);

    let conv1 = b.add_conv2d(silu0, s1w, Some(s1b), 2, 2, 1, 1, &[c1, s1, s1]);
    let bn1 = b.add_batch_norm(conv1, s1bm, s1bv, s1bw, s1bb, s1be, &[c1, s1, s1]);
    let sig1 = b.add_sigmoid(bn1, &[c1, s1, s1]);
    let silu1 = b.add_binary_mul(bn1, sig1, &[c1, s1, s1]);

    // SPPF: 3x MaxPool(k=5, s=1, p=2) + concat
    let pool1 = b.add_max_pool_2d(
        silu1,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &[c1, s1, s1],
    );
    let pool2 = b.add_max_pool_2d(
        pool1,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &[c1, s1, s1],
    );
    let pool3 = b.add_max_pool_2d(
        pool2,
        SPPF_POOL_K,
        SPPF_POOL_K,
        1,
        1,
        SPPF_POOL_PAD,
        SPPF_POOL_PAD,
        &[c1, s1, s1],
    );
    let sppf = b.add_concat(&[silu1, pool1, pool2, pool3], 0, &[sppf_c, s1, s1]);

    // Reduction: 1x1 conv (64 -> 16) -> BN -> SiLU
    let rw = b.add_input("red_w", &[c1, sppf_c, 1, 1]);
    let rb = b.add_input("red_b", &[c1]);
    let rbm = b.add_input("red_bm", &[c1]);
    let rbv = b.add_input("red_bv", &[c1]);
    let rbw = b.add_input("red_bw", &[c1]);
    let rbb = b.add_input("red_bb", &[c1]);
    let rbe = b.add_input("red_be", &[1]);

    let rc = b.add_conv2d(sppf, rw, Some(rb), 1, 1, 0, 0, &[c1, s1, s1]);
    let rbn = b.add_batch_norm(rc, rbm, rbv, rbw, rbb, rbe, &[c1, s1, s1]);
    let rsig = b.add_sigmoid(rbn, &[c1, s1, s1]);
    let out = b.add_binary_mul(rbn, rsig, &[c1, s1, s1]);

    b.build(out).expect("valid deep SPPF reduce kernel")
}

fn deep_sppf_reduce_bindings() -> Vec<TensorParamBinding> {
    let c0 = CONV_OUT_CHANNELS;
    let c1 = BOTTLENECK_CHANNELS;
    let sppf_c = c1 * 4;

    let mut v = vec![TensorParamBinding::Variable]; // image

    // Stage 0
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c0, IN_CHANNELS, 3, 3]),
        WEIGHT_MAG,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c0]),
        0.0f32,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c0]),
        0.0f32,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c0]),
        1.0f32,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c0]),
        1.0f32,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c0]),
        0.0f32,
    )));
    v.push(TensorParamBinding::ConstantScalar(1e-5));

    // Stage 1
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c1, c0, 3, 3]),
        WEIGHT_MAG,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c1]),
        0.0f32,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c1]),
        0.0f32,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c1]),
        1.0f32,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c1]),
        1.0f32,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c1]),
        0.0f32,
    )));
    v.push(TensorParamBinding::ConstantScalar(1e-5));

    // Reduction 1x1 + BN
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c1, sppf_c, 1, 1]),
        WEIGHT_MAG,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c1]),
        0.0f32,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c1]),
        0.0f32,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c1]),
        1.0f32,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c1]),
        1.0f32,
    )));
    v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[c1]),
        0.0f32,
    )));
    v.push(TensorParamBinding::ConstantScalar(1e-5));

    v
}

/// IBP through deep backbone -> SPPF -> reduction with BN + SiLU.
#[test]
fn test_deep_sppf_reduce_ibp() {
    let def = build_deep_sppf_reduce_kernel();
    let bindings = deep_sppf_reduce_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through deep SPPF reduce");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Deep SPPF reduce IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite());
    assert!(hi_max.is_finite());
}

/// CROWN through deep SPPF reduce: multi-stage + MaxPool chain + BN + SiLU.
#[test]
fn test_deep_sppf_reduce_crown() {
    let def = build_deep_sppf_reduce_kernel();
    let bindings = deep_sppf_reduce_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
    );
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Deep SPPF reduce CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 31. P3/P4/P5 feature pyramid with channel expansion
// ===========================================================================

/// Build a backbone that outputs P3, P4, P5 feature maps via progressive
/// downsampling, then concatenates them (flattened) for verification.
///
/// Input: `[3, 32, 32]` image.
/// Output: `[FPN_P3_C*FPN_P3_S*FPN_P3_S + FPN_P4_C*FPN_P4_S*FPN_P4_S + FPN_P5_C*FPN_P5_S*FPN_P5_S]`
///   flattened feature pyramid.
///
/// Architecture:
///   Stage 0: Conv(3, 8, 3, s=2, p=1) -> BN -> SiLU -> [8, 16, 16]
///   Stage 1: Conv(8, 16, 3, s=2, p=1) -> BN -> SiLU -> P3=[16, 8, 8]
///   Stage 2: Conv(16, 32, 3, s=2, p=1) -> BN -> SiLU -> P4=[32, 4, 4]
///   Stage 3: Conv(32, 64, 3, s=2, p=1) -> BN -> SiLU -> P5=[64, 2, 2]
///   Reshape each Pi to flat, concat
fn build_feature_pyramid_kernel() -> TensorKernelDef {
    let stages = [
        (IN_CHANNELS, DEEP_BB_C[0], DEEP_BB_S[0]),
        (DEEP_BB_C[0], FPN_P3_C, FPN_P3_S),
        (FPN_P3_C, FPN_P4_C, FPN_P4_S),
        (FPN_P4_C, FPN_P5_C, FPN_P5_S),
    ];

    let mut b = TensorBlockBuilder::new("deep_feature_pyramid");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let mut prev = input;
    let mut laterals = Vec::new();

    for (i, &(ci, co, so)) in stages.iter().enumerate() {
        let p = format!("s{i}");
        let shape = [co, so, so];

        let w = b.add_input(&format!("{p}_w"), &[co, ci, 3, 3]);
        let cb = b.add_input(&format!("{p}_b"), &[co]);
        let bm = b.add_input(&format!("{p}_bm"), &[co]);
        let bv = b.add_input(&format!("{p}_bv"), &[co]);
        let bw = b.add_input(&format!("{p}_bw"), &[co]);
        let bb = b.add_input(&format!("{p}_bb"), &[co]);
        let be = b.add_input(&format!("{p}_be"), &[1]);

        let conv = b.add_conv2d(prev, w, Some(cb), 2, 2, 1, 1, &shape);
        let bn = b.add_batch_norm(conv, bm, bv, bw, bb, be, &shape);
        let sig = b.add_sigmoid(bn, &shape);
        let silu = b.add_binary_mul(bn, sig, &shape);

        prev = silu;
        // Record P3, P4, P5 (stages 1, 2, 3)
        if i >= 1 {
            let flat_size = co * so * so;
            let flat = b.add_reshape(silu, &[flat_size]);
            laterals.push((flat, flat_size));
        }
    }

    // Concat flattened P3 + P4 + P5
    let total_size: usize = laterals.iter().map(|(_, s)| *s).sum();
    let flat_refs: Vec<_> = laterals.iter().map(|(node, _)| *node).collect();
    let out = b.add_concat(&flat_refs, 0, &[total_size]);

    b.build(out).expect("valid feature pyramid kernel")
}

fn feature_pyramid_bindings() -> Vec<TensorParamBinding> {
    let stages = [
        (IN_CHANNELS, DEEP_BB_C[0]),
        (DEEP_BB_C[0], FPN_P3_C),
        (FPN_P3_C, FPN_P4_C),
        (FPN_P4_C, FPN_P5_C),
    ];

    let mut v = vec![TensorParamBinding::Variable];
    for &(ci, co) in &stages {
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[co, ci, 3, 3]),
            WEIGHT_MAG,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[co]),
            0.0f32,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[co]),
            0.0f32,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[co]),
            1.0f32,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[co]),
            1.0f32,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[co]),
            0.0f32,
        )));
        v.push(TensorParamBinding::ConstantScalar(1e-5));
    }
    v
}

/// IBP through P3/P4/P5 feature pyramid backbone.
///
/// Verifies bounds propagation through progressive downsampling and
/// multi-scale feature extraction — the foundation for PAN neck input.
#[test]
fn test_feature_pyramid_ibp() {
    let def = build_feature_pyramid_kernel();
    let bindings = feature_pyramid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through feature pyramid");

    let total = FPN_P3_C * FPN_P3_S * FPN_P3_S
        + FPN_P4_C * FPN_P4_S * FPN_P4_S
        + FPN_P5_C * FPN_P5_S * FPN_P5_S;
    assert_eq!(output.lower_upper().0.shape(), &[total]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Feature pyramid IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite());
    assert!(hi_max.is_finite());
}

// ===========================================================================
// 32. Batch normalization accumulation: 5 sequential BN+SiLU stages
// ===========================================================================

/// Build 5 sequential BN -> SiLU blocks (no conv) to test BN bound
/// accumulation without the complexity of convolutions.
///
/// Input: `[16, 8, 8]` features (Variable).
/// Output: `[16, 8, 8]`.
fn build_bn_accumulation_kernel() -> TensorKernelDef {
    let c = BOTTLENECK_CHANNELS;
    let s = BOTTLENECK_SIZE;
    let shape = [c, s, s];
    let mut b = TensorBlockBuilder::new("deep_bn_accumulation");

    let mut prev = b.add_input("features", &shape);

    for i in 0..5 {
        let p = format!("bn{i}");
        let bm = b.add_input(&format!("{p}_bm"), &[c]);
        let bv = b.add_input(&format!("{p}_bv"), &[c]);
        let bw = b.add_input(&format!("{p}_bw"), &[c]);
        let bb = b.add_input(&format!("{p}_bb"), &[c]);
        let be = b.add_input(&format!("{p}_be"), &[1]);

        let bn = b.add_batch_norm(prev, bm, bv, bw, bb, be, &shape);
        let sig = b.add_sigmoid(bn, &shape);
        let silu = b.add_binary_mul(bn, sig, &shape);
        prev = silu;
    }

    b.build(prev).expect("valid BN accumulation kernel")
}

fn bn_accumulation_bindings() -> Vec<TensorParamBinding> {
    let c = BOTTLENECK_CHANNELS;
    let mut v = vec![TensorParamBinding::Variable];
    for _ in 0..5 {
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c]),
            0.0f32,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c]),
            1.0f32,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c]),
            1.0f32,
        )));
        v.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c]),
            0.0f32,
        )));
        v.push(TensorParamBinding::ConstantScalar(1e-5));
    }
    v
}

/// IBP through 5 sequential BN+SiLU: tests BN bound accumulation depth.
#[test]
fn test_bn_accumulation_ibp() {
    let def = build_bn_accumulation_kernel();
    let bindings = bn_accumulation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        2.0,
    );

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through BN accumulation");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("BN accumulation 5-stage IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite());
    assert!(hi_max.is_finite());
}

/// CROWN through 5 sequential BN+SiLU: CROWN linearizes each sigmoid.
#[test]
fn test_bn_accumulation_crown() {
    let def = build_bn_accumulation_kernel();
    let bindings = bn_accumulation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
        2.0,
    );

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE],
    );
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("BN accumulation 5-stage CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 34. Full PAN top-down path: P5 -> lateral conv -> upsample -> merge P4 ->
//     lateral conv -> upsample -> merge P3
// ===========================================================================

/// Build full PAN top-down fusion path across 3 scales.
///
/// Input (Variable): `[FPN_P5_C, FPN_P5_S, FPN_P5_S]` (P5 features).
/// Constant: P4 and P3 lateral features.
/// Output: `[FPN_P3_C * 2, FPN_P3_S, FPN_P3_S]` fused P3 features.
///
/// Top-down:
///   P5 -> 1x1 conv (64->32) -> reshape (upsample 2x to 4x4) -> concat P4
///   -> BN -> SiLU -> 1x1 conv (64->16) -> reshape (upsample 2x to 8x8)
///   -> concat P3
fn build_pan_full_topdown_kernel() -> TensorKernelDef {
    let c5 = FPN_P5_C; // 64
    let c4 = FPN_P4_C; // 32
    let c3 = FPN_P3_C; // 16
    let s5 = FPN_P5_S; // 2
    let s4 = FPN_P4_S; // 4
    let s3 = FPN_P3_S; // 8

    let mut b = TensorBlockBuilder::new("deep_pan_full_topdown");

    let p5 = b.add_input("p5_feat", &[c5, s5, s5]);
    let p4_lat = b.add_input("p4_lateral", &[c4, s4, s4]);
    let p3_lat = b.add_input("p3_lateral", &[c3, s3, s3]);

    // P5 -> 1x1 conv to reduce channels (64 -> 32)
    let td1_w = b.add_input("td1_w", &[c4, c5, 1, 1]);
    let td1_b = b.add_input("td1_b", &[c4]);
    let td1_conv = b.add_conv2d(p5, td1_w, Some(td1_b), 1, 1, 0, 0, &[c4, s5, s5]);

    // Upsample 2x: reshape [32, 2, 2] -> [32/4, 4, 4] = [8, 4, 4]
    // Model nearest-neighbor upsample as reshape for verification
    let up1_c = c4 * s5 * s5 / (s4 * s4); // 32*4/16 = 8
    let up1 = b.add_reshape(td1_conv, &[up1_c, s4, s4]);

    // Concat with P4 lateral: [8, 4, 4] + [32, 4, 4] = [40, 4, 4]
    let cat1_c = up1_c + c4;
    let cat1 = b.add_concat(&[up1, p4_lat], 0, &[cat1_c, s4, s4]);

    // Fuse: 1x1 conv (40 -> 16) -> BN -> SiLU
    let f1_w = b.add_input("f1_w", &[c3, cat1_c, 1, 1]);
    let f1_b = b.add_input("f1_b", &[c3]);
    let f1_bm = b.add_input("f1_bm", &[c3]);
    let f1_bv = b.add_input("f1_bv", &[c3]);
    let f1_bw = b.add_input("f1_bw", &[c3]);
    let f1_bb = b.add_input("f1_bb", &[c3]);
    let f1_be = b.add_input("f1_be", &[1]);

    let f1_conv = b.add_conv2d(cat1, f1_w, Some(f1_b), 1, 1, 0, 0, &[c3, s4, s4]);
    let f1_bn = b.add_batch_norm(f1_conv, f1_bm, f1_bv, f1_bw, f1_bb, f1_be, &[c3, s4, s4]);
    let f1_sig = b.add_sigmoid(f1_bn, &[c3, s4, s4]);
    let f1_silu = b.add_binary_mul(f1_bn, f1_sig, &[c3, s4, s4]);

    // Upsample 2x: reshape [16, 4, 4] -> [4, 8, 8]
    let up2_c = c3 * s4 * s4 / (s3 * s3); // 16*16/64 = 4
    let up2 = b.add_reshape(f1_silu, &[up2_c, s3, s3]);

    // Concat with P3 lateral: [4, 8, 8] + [16, 8, 8] = [20, 8, 8]
    let cat2_c = up2_c + c3;
    let out = b.add_concat(&[up2, p3_lat], 0, &[cat2_c, s3, s3]);

    b.build(out).expect("valid PAN full top-down kernel")
}

fn pan_full_topdown_bindings() -> Vec<TensorParamBinding> {
    let c5 = FPN_P5_C;
    let c4 = FPN_P4_C;
    let c3 = FPN_P3_C;
    let s4 = FPN_P4_S;
    let s3 = FPN_P3_S;
    let up1_c = c4 * FPN_P5_S * FPN_P5_S / (s4 * s4);
    let cat1_c = up1_c + c4;

    vec![
        TensorParamBinding::Variable, // p5_feat
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4, s4, s4]), 0.5f32)), // p4_lateral
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c3, s3, s3]), 0.5f32)), // p3_lateral
        // td1 conv
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4, c5, 1, 1]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 0.0f32)),
        // f1 conv + BN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c3, cat1_c, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c3]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c3]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c3]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c3]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c3]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

/// IBP through full PAN top-down path: P5 -> lateral -> upsample -> P4 merge -> P3 merge.
#[test]
fn test_pan_full_topdown_ibp() {
    let def = build_pan_full_topdown_kernel();
    let bindings = pan_full_topdown_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FPN_P5_C, FPN_P5_S, FPN_P5_S], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN full top-down");

    let up2_c = FPN_P3_C * FPN_P4_S * FPN_P4_S / (FPN_P3_S * FPN_P3_S);
    let out_c = up2_c + FPN_P3_C;
    assert_eq!(output.lower_upper().0.shape(), &[out_c, FPN_P3_S, FPN_P3_S]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PAN full top-down IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite());
    assert!(hi_max.is_finite());
}

// ===========================================================================
// 35. PAN full bottom-up path: P3 -> downsample -> merge P4 -> downsample -> P5
// ===========================================================================

/// Build full PAN bottom-up path: P3 features progressively downsampled
/// and merged with P4 and P5.
///
/// Input (Variable): `[FPN_P3_C, FPN_P3_S, FPN_P3_S]` (P3 features).
/// Constant: P4, P5 features.
/// Output: `[FPN_P5_C * 2, FPN_P5_S, FPN_P5_S]` merged P5 output.
fn build_pan_full_bottomup_kernel() -> TensorKernelDef {
    let c3 = FPN_P3_C; // 16
    let c4 = FPN_P4_C; // 32
    let c5 = FPN_P5_C; // 64
    let s3 = FPN_P3_S; // 8
    let s4 = FPN_P4_S; // 4
    let s5 = FPN_P5_S; // 2

    let mut b = TensorBlockBuilder::new("deep_pan_full_bottomup");

    let p3 = b.add_input("p3_feat", &[c3, s3, s3]);
    let p4_lat = b.add_input("p4_lateral", &[c4, s4, s4]);
    let p5_lat = b.add_input("p5_lateral", &[c5, s5, s5]);

    // P3 -> stride-2 conv to downsample (16, 8, 8) -> (c4, 4, 4)
    let d1_w = b.add_input("d1_w", &[c4, c3, 3, 3]);
    let d1_b = b.add_input("d1_b", &[c4]);
    let d1_bm = b.add_input("d1_bm", &[c4]);
    let d1_bv = b.add_input("d1_bv", &[c4]);
    let d1_bw = b.add_input("d1_bw", &[c4]);
    let d1_bb = b.add_input("d1_bb", &[c4]);
    let d1_be = b.add_input("d1_be", &[1]);

    let d1_conv = b.add_conv2d(p3, d1_w, Some(d1_b), 2, 2, 1, 1, &[c4, s4, s4]);
    let d1_bn = b.add_batch_norm(d1_conv, d1_bm, d1_bv, d1_bw, d1_bb, d1_be, &[c4, s4, s4]);
    let d1_sig = b.add_sigmoid(d1_bn, &[c4, s4, s4]);
    let d1_silu = b.add_binary_mul(d1_bn, d1_sig, &[c4, s4, s4]);

    // Concat with P4: [32, 4, 4] + [32, 4, 4] = [64, 4, 4]
    let cat1_c = c4 * 2;
    let cat1 = b.add_concat(&[d1_silu, p4_lat], 0, &[cat1_c, s4, s4]);

    // Fuse P4: 1x1 conv (64 -> 32) -> BN -> SiLU
    let f1_w = b.add_input("f1_w", &[c4, cat1_c, 1, 1]);
    let f1_b = b.add_input("f1_b", &[c4]);
    let f1_bm = b.add_input("f1_bm", &[c4]);
    let f1_bv = b.add_input("f1_bv", &[c4]);
    let f1_bw = b.add_input("f1_bw", &[c4]);
    let f1_bb = b.add_input("f1_bb", &[c4]);
    let f1_be = b.add_input("f1_be", &[1]);

    let f1_conv = b.add_conv2d(cat1, f1_w, Some(f1_b), 1, 1, 0, 0, &[c4, s4, s4]);
    let f1_bn = b.add_batch_norm(f1_conv, f1_bm, f1_bv, f1_bw, f1_bb, f1_be, &[c4, s4, s4]);
    let f1_sig = b.add_sigmoid(f1_bn, &[c4, s4, s4]);
    let f1_silu = b.add_binary_mul(f1_bn, f1_sig, &[c4, s4, s4]);

    // Downsample to P5: stride-2 conv (32, 4, 4) -> (64, 2, 2)
    let d2_w = b.add_input("d2_w", &[c5, c4, 3, 3]);
    let d2_b = b.add_input("d2_b", &[c5]);
    let d2_bm = b.add_input("d2_bm", &[c5]);
    let d2_bv = b.add_input("d2_bv", &[c5]);
    let d2_bw = b.add_input("d2_bw", &[c5]);
    let d2_bb = b.add_input("d2_bb", &[c5]);
    let d2_be = b.add_input("d2_be", &[1]);

    let d2_conv = b.add_conv2d(f1_silu, d2_w, Some(d2_b), 2, 2, 1, 1, &[c5, s5, s5]);
    let d2_bn = b.add_batch_norm(d2_conv, d2_bm, d2_bv, d2_bw, d2_bb, d2_be, &[c5, s5, s5]);
    let d2_sig = b.add_sigmoid(d2_bn, &[c5, s5, s5]);
    let d2_silu = b.add_binary_mul(d2_bn, d2_sig, &[c5, s5, s5]);

    // Concat with P5 lateral: [64, 2, 2] + [64, 2, 2] = [128, 2, 2]
    let out_c = c5 * 2;
    let out = b.add_concat(&[d2_silu, p5_lat], 0, &[out_c, s5, s5]);

    b.build(out).expect("valid PAN full bottom-up kernel")
}

fn pan_full_bottomup_bindings() -> Vec<TensorParamBinding> {
    let c3 = FPN_P3_C;
    let c4 = FPN_P4_C;
    let c5 = FPN_P5_C;
    let s4 = FPN_P4_S;
    let s5 = FPN_P5_S;
    let cat1_c = c4 * 2;

    vec![
        TensorParamBinding::Variable, // p3_feat
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4, s4, s4]), 0.5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5, s5, s5]), 0.5f32)),
        // d1 conv + BN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4, c3, 3, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // f1 conv + BN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c4, cat1_c, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c4]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // d2 conv + BN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5, c4, 3, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

/// IBP through full PAN bottom-up: P3 downsample -> P4 merge -> P5 merge.
#[test]
fn test_pan_full_bottomup_ibp() {
    let def = build_pan_full_bottomup_kernel();
    let bindings = pan_full_bottomup_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FPN_P3_C, FPN_P3_S, FPN_P3_S], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN full bottom-up");

    let out_c = FPN_P5_C * 2;
    assert_eq!(output.lower_upper().0.shape(), &[out_c, FPN_P5_S, FPN_P5_S]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PAN full bottom-up IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite());
    assert!(hi_max.is_finite());
}

/// CROWN through PAN full bottom-up: downsample + concat + fuse + SiLU chain.
#[test]
fn test_pan_full_bottomup_crown() {
    let def = build_pan_full_bottomup_kernel();
    let bindings = pan_full_bottomup_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FPN_P3_C, FPN_P3_S, FPN_P3_S], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let out_c = FPN_P5_C * 2;
    assert_eq!(output.lower_upper().0.shape(), &[out_c, FPN_P5_S, FPN_P5_S]);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PAN full bottom-up CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 37. Bidirectional PAN: top-down + bottom-up in single graph
// ===========================================================================

/// Build a simplified bidirectional PAN: top-down then bottom-up in sequence.
///
/// Input (Variable): `[FPN_P5_C, FPN_P5_S, FPN_P5_S]` P5 features.
/// Constant: P3 features for the bottom-up path.
/// Output: `[FPN_P5_C, FPN_P5_S, FPN_P5_S]` refined P5 features.
///
/// Architecture:
///   Top-down: P5 -> 1x1 conv reduce -> reshape as upsample
///   Bottom-up: upsample -> stride-2 conv -> BN -> SiLU -> concat P5 -> 1x1 reduce
fn build_bidirectional_pan_kernel() -> TensorKernelDef {
    let c5 = FPN_P5_C; // 64
    let s5 = FPN_P5_S; // 2
    let s_mid = s5 * 2; // 4

    let mut b = TensorBlockBuilder::new("deep_bidirectional_pan");

    let p5 = b.add_input("p5_feat", &[c5, s5, s5]);

    // Top-down: 1x1 conv (64 -> 16) then reshape to upsample [16, 2, 2] -> [4, 4, 4]
    let td_w = b.add_input("td_w", &[c5 / 4, c5, 1, 1]);
    let td_b = b.add_input("td_b", &[c5 / 4]);
    let td_conv = b.add_conv2d(p5, td_w, Some(td_b), 1, 1, 0, 0, &[c5 / 4, s5, s5]);

    let mid_c = (c5 / 4) * s5 * s5 / (s_mid * s_mid);
    let td_up = b.add_reshape(td_conv, &[mid_c, s_mid, s_mid]);

    // Bottom-up: stride-2 conv (mid_c, 4, 4) -> (c5, 2, 2) -> BN -> SiLU
    let bu_w = b.add_input("bu_w", &[c5, mid_c, 3, 3]);
    let bu_b = b.add_input("bu_b", &[c5]);
    let bu_bm = b.add_input("bu_bm", &[c5]);
    let bu_bv = b.add_input("bu_bv", &[c5]);
    let bu_bw = b.add_input("bu_bw", &[c5]);
    let bu_bb = b.add_input("bu_bb", &[c5]);
    let bu_be = b.add_input("bu_be", &[1]);

    let bu_conv = b.add_conv2d(td_up, bu_w, Some(bu_b), 2, 2, 1, 1, &[c5, s5, s5]);
    let bu_bn = b.add_batch_norm(bu_conv, bu_bm, bu_bv, bu_bw, bu_bb, bu_be, &[c5, s5, s5]);
    let bu_sig = b.add_sigmoid(bu_bn, &[c5, s5, s5]);
    let bu_silu = b.add_binary_mul(bu_bn, bu_sig, &[c5, s5, s5]);

    // Residual merge with original P5
    let out = b.add_binary_add(bu_silu, p5, &[c5, s5, s5]);

    b.build(out).expect("valid bidirectional PAN kernel")
}

fn bidirectional_pan_bindings() -> Vec<TensorParamBinding> {
    let c5 = FPN_P5_C;
    let s5 = FPN_P5_S;
    let s_mid = s5 * 2;
    let mid_c = (c5 / 4) * s5 * s5 / (s_mid * s_mid);

    vec![
        TensorParamBinding::Variable, // p5_feat
        // td conv
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c5 / 4, c5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5 / 4]), 0.0f32)),
        // bu conv + BN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c5, mid_c, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c5]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

/// IBP through bidirectional PAN: top-down upsample + bottom-up downsample + residual.
#[test]
fn test_bidirectional_pan_ibp() {
    let def = build_bidirectional_pan_kernel();
    let bindings = bidirectional_pan_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FPN_P5_C, FPN_P5_S, FPN_P5_S], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through bidirectional PAN");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[FPN_P5_C, FPN_P5_S, FPN_P5_S]
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Bidirectional PAN IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite());
    assert!(hi_max.is_finite());
}

/// CROWN through bidirectional PAN.
#[test]
fn test_bidirectional_pan_crown() {
    let def = build_bidirectional_pan_kernel();
    let bindings = bidirectional_pan_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FPN_P5_C, FPN_P5_S, FPN_P5_S], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[FPN_P5_C, FPN_P5_S, FPN_P5_S]
    );
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Bidirectional PAN CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 39. Multi-scale feature alignment: reshape + linear projection
// ===========================================================================

/// Build a multi-scale alignment block that projects P3, P4, P5 features
/// to a common channel width before merging.
///
/// Input (Variable): `[FPN_P5_C, FPN_P5_S, FPN_P5_S]` (P5).
/// Output: `[common_c, total_anchors]` aligned features.
///
/// Each scale is projected to `common_c` channels via 1x1 conv, flattened,
/// and concatenated.
fn build_multiscale_alignment_kernel() -> TensorKernelDef {
    let c5 = FPN_P5_C; // 64
    let s5 = FPN_P5_S; // 2
    let common_c: usize = 16;
    let anchors = s5 * s5; // 4

    let mut b = TensorBlockBuilder::new("deep_multiscale_align");

    let p5 = b.add_input("p5_feat", &[c5, s5, s5]);

    // Project P5 to common channels: 1x1 conv (64 -> 16)
    let pw = b.add_input("proj_w", &[common_c, c5, 1, 1]);
    let pb = b.add_input("proj_b", &[common_c]);
    let proj = b.add_conv2d(p5, pw, Some(pb), 1, 1, 0, 0, &[common_c, s5, s5]);

    // BN -> SiLU
    let bm = b.add_input("proj_bm", &[common_c]);
    let bv = b.add_input("proj_bv", &[common_c]);
    let bw = b.add_input("proj_bw", &[common_c]);
    let bb = b.add_input("proj_bb", &[common_c]);
    let be = b.add_input("proj_be", &[1]);

    let bn = b.add_batch_norm(proj, bm, bv, bw, bb, be, &[common_c, s5, s5]);
    let sig = b.add_sigmoid(bn, &[common_c, s5, s5]);
    let silu = b.add_binary_mul(bn, sig, &[common_c, s5, s5]);

    // Reshape to [common_c, anchors] for downstream heads
    let out = b.add_reshape(silu, &[common_c, anchors]);

    b.build(out).expect("valid multi-scale alignment kernel")
}

fn multiscale_alignment_bindings() -> Vec<TensorParamBinding> {
    let c5 = FPN_P5_C;
    let common_c: usize = 16;

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[common_c, c5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[common_c]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[common_c]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[common_c]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[common_c]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[common_c]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

/// IBP through multi-scale feature alignment.
#[test]
fn test_multiscale_alignment_ibp() {
    let def = build_multiscale_alignment_kernel();
    let bindings = multiscale_alignment_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FPN_P5_C, FPN_P5_S, FPN_P5_S], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-scale alignment");

    let common_c: usize = 16;
    let anchors = FPN_P5_S * FPN_P5_S;
    assert_eq!(output.lower_upper().0.shape(), &[common_c, anchors]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-scale alignment IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite());
    assert!(hi_max.is_finite());
}

// ===========================================================================
// 40. Anchor-free detection head at 3 scales with sigmoid cls
// ===========================================================================

/// Build anchor-free detection head: linear projection -> sigmoid at 3 scales.
///
/// Input (Variable): concatenated features from 3 scales flattened.
/// Output: `[total_anchors, NUM_CLASSES]` class probabilities in [0, 1].
///
/// Tests that sigmoid guarantees [0,1] output across all scales.
fn build_anchor_free_3scale_head_kernel() -> TensorKernelDef {
    let s3 = FPN_P3_S; // 8
    let s4 = FPN_P4_S; // 4
    let s5 = FPN_P5_S; // 2
    let anchors3 = s3 * s3; // 64
    let anchors4 = s4 * s4; // 16
    let anchors5 = s5 * s5; // 4
    let total = anchors3 + anchors4 + anchors5; // 84
    let hidden = AF_HEAD_HIDDEN; // 32
    let cls = NUM_CLASSES; // 10

    let mut b = TensorBlockBuilder::new("deep_anchor_free_3scale");

    let input = b.add_input("features", &[total, hidden]);

    // Linear projection: [84, 32] x [10, 32]^T -> [84, 10]
    let w = b.add_input("cls_w", &[cls, hidden]);
    let bias = b.add_input("cls_b", &[cls]);

    let logits = b.add_matmul(input, w, true, None, &[total, cls]);
    let bias_bc = b.add_broadcast(bias, &[total, cls]);
    let biased = b.add_binary_add(logits, bias_bc, &[total, cls]);
    let out = b.add_sigmoid(biased, &[total, cls]);

    b.build(out).expect("valid anchor-free 3-scale head kernel")
}

fn anchor_free_3scale_head_bindings() -> Vec<TensorParamBinding> {
    let hidden = AF_HEAD_HIDDEN;
    let cls = NUM_CLASSES;

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[cls, hidden]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[cls]), 0.0f32)),
    ]
}

/// IBP through anchor-free 3-scale head: sigmoid guarantees [0, 1].
#[test]
fn test_anchor_free_3scale_head_ibp() {
    let s3 = FPN_P3_S;
    let s4 = FPN_P4_S;
    let s5 = FPN_P5_S;
    let total = s3 * s3 + s4 * s4 + s5 * s5;

    let def = build_anchor_free_3scale_head_kernel();
    let bindings = anchor_free_3scale_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[total, AF_HEAD_HIDDEN], 3.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through anchor-free 3-scale head");

    assert_eq!(output.lower_upper().0.shape(), &[total, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Anchor-free 3-scale head IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "sigmoid lb >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "sigmoid ub <= 1, got {hi_max}");
}

/// CROWN through anchor-free 3-scale head.
#[test]
fn test_anchor_free_3scale_head_crown() {
    let s3 = FPN_P3_S;
    let s4 = FPN_P4_S;
    let s5 = FPN_P5_S;
    let total = s3 * s3 + s4 * s4 + s5 * s5;

    let def = build_anchor_free_3scale_head_kernel();
    let bindings = anchor_free_3scale_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[total, AF_HEAD_HIDDEN], 3.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[total, NUM_CLASSES]);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Anchor-free 3-scale head CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 42. DFL regression full pipeline: conv -> reshape -> softmax -> bins matmul
// ===========================================================================

/// Build full DFL regression pipeline with conv feature extraction.
///
/// Input (Variable): `[AF_HEAD_HIDDEN, FPN_P5_S, FPN_P5_S]` neck features.
/// Output: `[anchors * 4, 1]` DFL-decoded box coordinates.
///
/// Architecture:
///   Conv(32, 64, 3, s=1, p=1) -> BN -> SiLU
///   Conv(64, 4*DFL_BINS, 1, s=1, p=0)
///   Reshape to [anchors*4, DFL_BINS]
///   Softmax(dim=1) -> matmul with bins -> [anchors*4, 1]
fn build_dfl_full_pipeline_kernel() -> TensorKernelDef {
    let c_in = AF_HEAD_HIDDEN; // 32
    let c_hidden = FPN_P5_C; // 64
    let s = FPN_P5_S; // 2
    let anchors = s * s; // 4
    let reg_out = 4 * DFL_BINS; // 64
    let flat_len = anchors * 4; // 16

    let mut b = TensorBlockBuilder::new("deep_dfl_full_pipeline");

    let input = b.add_input("features", &[c_in, s, s]);

    // Conv(32, 64, 3, s=1, p=1) -> BN -> SiLU
    let c1w = b.add_input("c1_w", &[c_hidden, c_in, 3, 3]);
    let c1b = b.add_input("c1_b", &[c_hidden]);
    let c1bm = b.add_input("c1_bm", &[c_hidden]);
    let c1bv = b.add_input("c1_bv", &[c_hidden]);
    let c1bw = b.add_input("c1_bw", &[c_hidden]);
    let c1bb = b.add_input("c1_bb", &[c_hidden]);
    let c1be = b.add_input("c1_be", &[1]);

    let conv1 = b.add_conv2d(input, c1w, Some(c1b), 1, 1, 1, 1, &[c_hidden, s, s]);
    let bn1 = b.add_batch_norm(conv1, c1bm, c1bv, c1bw, c1bb, c1be, &[c_hidden, s, s]);
    let sig1 = b.add_sigmoid(bn1, &[c_hidden, s, s]);
    let silu1 = b.add_binary_mul(bn1, sig1, &[c_hidden, s, s]);

    // Conv(64, 4*16=64, 1, s=1, p=0)
    let c2w = b.add_input("c2_w", &[reg_out, c_hidden, 1, 1]);
    let c2b = b.add_input("c2_b", &[reg_out]);
    let conv2 = b.add_conv2d(silu1, c2w, Some(c2b), 1, 1, 0, 0, &[reg_out, s, s]);

    // Reshape to [anchors*4, DFL_BINS]
    let reshaped = b.add_reshape(conv2, &[flat_len, DFL_BINS]);

    // DFL decode: softmax + matmul with bins
    let bins = b.add_input("dfl_bins", &[DFL_BINS, 1]);
    let probs = b.add_softmax(reshaped, 1, &[flat_len, DFL_BINS]);
    let out = b.add_matmul(probs, bins, false, None, &[flat_len, 1]);

    b.build(out).expect("valid DFL full pipeline kernel")
}

fn dfl_full_pipeline_bindings() -> Vec<TensorParamBinding> {
    let c_in = AF_HEAD_HIDDEN;
    let c_hidden = FPN_P5_C;
    let reg_out = 4 * DFL_BINS;
    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();

    vec![
        TensorParamBinding::Variable,
        // conv1 + BN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c_hidden, c_in, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_hidden]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_hidden]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_hidden]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_hidden]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c_hidden]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        // conv2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[reg_out, c_hidden, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[reg_out]), 0.0f32)),
        // bins
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins"),
        ),
    ]
}

/// IBP through full DFL regression pipeline: conv -> BN -> SiLU -> conv -> softmax -> bins.
#[test]
fn test_dfl_full_pipeline_ibp() {
    let def = build_dfl_full_pipeline_kernel();
    let bindings = dfl_full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[AF_HEAD_HIDDEN, FPN_P5_S, FPN_P5_S], 2.0);

    let anchors = FPN_P5_S * FPN_P5_S;
    let flat_len = anchors * 4;

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DFL full pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[flat_len, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DFL full pipeline IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite());
    assert!(hi_max.is_finite());
    // DFL output is a weighted sum of bins [0, DFL_BINS-1]; bounds should reflect this.
    assert!(
        lo_min >= -1.0,
        "DFL output lower bound should be >= -1, got {lo_min}"
    );
    assert!(
        hi_max <= DFL_BINS as f32,
        "DFL output should be <= {DFL_BINS}, got {hi_max}",
    );
}

/// CROWN through full DFL pipeline: linearizes SiLU and softmax.
#[test]
fn test_dfl_full_pipeline_crown() {
    let def = build_dfl_full_pipeline_kernel();
    let bindings = dfl_full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[AF_HEAD_HIDDEN, FPN_P5_S, FPN_P5_S], 2.0);

    let anchors = FPN_P5_S * FPN_P5_S;
    let flat_len = anchors * 4;

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[flat_len, 1]);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DFL full pipeline CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 44. NMS stability: wide-range logits through sigmoid composition
// ===========================================================================

/// Build NMS stability test: matmul with random-magnitude weights -> sigmoid.
///
/// Input (Variable): `[NUM_ANCHORS, AF_HEAD_HIDDEN]` raw features.
/// Output: `[NUM_ANCHORS, NUM_CLASSES]` class scores in [0, 1].
///
/// Tests that sigmoid output [0, 1] property holds even with large
/// intermediate logit values from the linear projection.
fn build_nms_stability_kernel() -> TensorKernelDef {
    let n = NUM_ANCHORS; // 16
    let h = AF_HEAD_HIDDEN; // 32
    let cls = NUM_CLASSES; // 10

    let mut b = TensorBlockBuilder::new("deep_nms_stability");

    let input = b.add_input("features", &[n, h]);
    let w = b.add_input("cls_w", &[cls, h]);
    let bias = b.add_input("cls_b", &[cls]);

    let logits = b.add_matmul(input, w, true, None, &[n, cls]);
    let bias_bc = b.add_broadcast(bias, &[n, cls]);
    let biased = b.add_binary_add(logits, bias_bc, &[n, cls]);
    let out = b.add_sigmoid(biased, &[n, cls]);

    b.build(out).expect("valid NMS stability kernel")
}

fn nms_stability_bindings() -> Vec<TensorParamBinding> {
    let h = AF_HEAD_HIDDEN;
    let cls = NUM_CLASSES;

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[cls, h]), 0.1f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[cls]), 0.0f32)),
    ]
}

/// IBP through NMS stability with wide input range [-50, 50].
///
/// Even with extreme input magnitudes, sigmoid guarantees [0, 1] output.
#[test]
fn test_nms_stability_wide_ibp() {
    let def = build_nms_stability_kernel();
    let bindings = nms_stability_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, AF_HEAD_HIDDEN], 50.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through NMS stability");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_ANCHORS, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("NMS stability wide-range IBP [-50,50]: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "sigmoid lb >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "sigmoid ub <= 1, got {hi_max}");
}

// ===========================================================================
// 45. Verify-and-record: deep C2f split 3-bottleneck
// ===========================================================================

/// Verify and record deep C2f split 3-bottleneck.
#[test]
fn test_c2f_split_three_bottleneck_verify_and_record() {
    let def = build_c2f_split_three_bottleneck_kernel();
    let bindings = c2f_split_three_bottleneck_bindings();
    let c = DEEP_BB_C[1];
    let s = DEEP_BB_S[1];
    let input = uniform_bounds(&[c, s, s], 2.0);

    let result = verify_and_assert(&def, &bindings, &input, "deep_c2f_split_3bn");
    assert_eq!(result.num_variables, 1, "single Variable input");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[c, s, s]);
}

// ===========================================================================
// 46. Verify-and-record: PAN full bottom-up
// ===========================================================================

/// Verify and record PAN full bottom-up path.
#[test]
fn test_pan_full_bottomup_verify_and_record() {
    let def = build_pan_full_bottomup_kernel();
    let bindings = pan_full_bottomup_bindings();
    let input = uniform_bounds(&[FPN_P3_C, FPN_P3_S, FPN_P3_S], 2.0);

    let result = verify_and_assert(&def, &bindings, &input, "deep_pan_full_bottomup");
    assert_eq!(result.num_variables, 1, "single Variable input");
    let out_c = FPN_P5_C * 2;
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[out_c, FPN_P5_S, FPN_P5_S]);
}

// ===========================================================================
// 47. Verify-and-record: DFL full pipeline
// ===========================================================================

/// Verify and record DFL full pipeline.
#[test]
fn test_dfl_full_pipeline_verify_and_record() {
    let def = build_dfl_full_pipeline_kernel();
    let bindings = dfl_full_pipeline_bindings();
    let input = uniform_bounds(&[AF_HEAD_HIDDEN, FPN_P5_S, FPN_P5_S], 2.0);

    let anchors = FPN_P5_S * FPN_P5_S;
    let flat_len = anchors * 4;

    let result = verify_and_assert(&def, &bindings, &input, "deep_dfl_full_pipeline");
    assert_eq!(result.num_variables, 1, "single Variable input");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[flat_len, 1]);
}
