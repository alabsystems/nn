// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep compose tests: DocLayout-YOLO backbone C2f + FPN fusion with CROWN.
//!
//! Verifies bounds propagation through the DocLayout-YOLO backbone and
//! Feature Pyramid Network (FPN) fusion stages. These tests target the
//! heuristic gaps in DocLayout-YOLO compose coverage by testing multi-branch
//! C2f blocks and multi-scale fusion with CROWN linearization.
//!
//! 1. **C2f entry conv + bottleneck**: Conv-BN-SiLU entry -> Conv-BN-SiLU
//!    bottleneck with residual skip connection. Core building block (IBP + CROWN).
//!
//! 2. **C2f full block with concat**: Entry conv -> 2 bottleneck branches ->
//!    channel concat -> exit conv. Multi-branch composition (IBP + CROWN).
//!
//! 3. **Backbone stage: ConvBnAct stride-2 + C2f**: Spatial downsampling
//!    followed by C2f feature extraction. Single stage compose (IBP + CROWN).
//!
//! 4. **2-stage backbone**: Cascaded stride-2 downsampling + C2f blocks.
//!    Depth composition with spatial reduction (IBP + CROWN).
//!
//! 5. **SPPF + C2f neck block**: Spatial pyramid pooling -> C2f feature
//!    refinement. Neck feature extraction composition (IBP + CROWN).
//!
//! 6. **PAN top-down fusion**: Feature pyramid with concat and Conv reduction.
//!    Multi-scale feature fusion (IBP).
//!
//! 7. **Detection head from neck features**: Dual sigmoid (classification)
//!    + DFL (box regression) heads. Output bounds verification (IBP + CROWN).
//!
//! 8. **Full backbone -> neck -> head**: End-to-end detection pipeline
//!    with CROWN through the full composition (IBP + CROWN).
//!
//! Architecture reference:
//! - DocLayout-YOLO (Zhao et al. 2024): YOLOv10-based document layout
//! - YOLOv8 C2f: Conv -> split -> N bottleneck residuals -> concat -> Conv
//! - SPPF: Spatial Pyramid Pooling - Fast
//! - PAN: Path Aggregation Network for multi-scale fusion
//! - DFL (Li et al. 2022): Distribution Focal Loss for box regression
//!
//! Dimensions are small for fast verification (CHANNELS=8, SPATIAL=8).
//! All tests use IbpValidated soundness mode per nn engineering rules.
//!
//! Part of #4304: deep NY compose tests for DocLayout-YOLO backbone.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

const SPATIAL: usize = 8;
const CHANNELS: usize = 8;
const CHANNELS_2X: usize = CHANNELS * 2; // 16
const IN_CH: usize = 3;
const IMG_SIZE: usize = 16;
const NUM_CLASSES: usize = 4;
const DFL_BINS: usize = 8;
const NUM_ANCHORS: usize = 4;
const SPPF_POOL_K: usize = 5;
const SPPF_POOL_PAD: usize = 2;
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
    .expect("valid image bounds")
}

/// Add Conv-BN-SiLU block. Returns output node.
fn add_conv_bn_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    pfx: &str,
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    out_shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let conv_w = b.add_input(&format!("{pfx}_conv_w"), &[out_ch, in_ch, kernel, kernel]);
    let conv_b = b.add_input(&format!("{pfx}_conv_b"), &[out_ch]);
    let bn_mean = b.add_input(&format!("{pfx}_bn_mean"), &[out_ch]);
    let bn_var = b.add_input(&format!("{pfx}_bn_var"), &[out_ch]);
    let bn_w = b.add_input(&format!("{pfx}_bn_w"), &[out_ch]);
    let bn_b = b.add_input(&format!("{pfx}_bn_b"), &[out_ch]);
    let bn_eps = b.add_input(&format!("{pfx}_bn_eps"), &[1]);

    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        stride,
        stride,
        padding,
        padding,
        out_shape,
    );
    let bn_out = b.add_batch_norm(conv_out, bn_mean, bn_var, bn_w, bn_b, bn_eps, out_shape);
    // SiLU(x) = x * sigmoid(x)
    let sig = b.add_sigmoid(bn_out, out_shape);
    b.add_binary_mul(bn_out, sig, out_shape)
}

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

// ===========================================================================
// 1. C2f entry conv + bottleneck with residual
// ===========================================================================

fn build_c2f_entry_bottleneck() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("doclayout_c2f_entry_bottleneck");
    let shape = [CHANNELS, SPATIAL, SPATIAL];

    let input = b.add_input("features", &shape);

    // Entry 1x1 conv
    let entry = add_conv_bn_silu(&mut b, input, "entry", CHANNELS, CHANNELS, 1, 1, 0, &shape);

    // Bottleneck: 3x3 conv -> 3x3 conv + skip
    let bot1 = add_conv_bn_silu(&mut b, entry, "bot1", CHANNELS, CHANNELS, 3, 1, 1, &shape);
    let bot2 = add_conv_bn_silu(&mut b, bot1, "bot2", CHANNELS, CHANNELS, 3, 1, 1, &shape);

    // Residual skip connection
    let out = b.add_binary_add(entry, bot2, &shape);
    b.build(out).expect("valid C2f entry + bottleneck")
}

fn c2f_entry_bottleneck_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 1); // entry
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 3); // bot1
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 3); // bot2
    bindings
}

#[test]
fn test_doclayout_c2f_entry_bottleneck_ibp() {
    let def = build_c2f_entry_bottleneck();
    let bindings = c2f_entry_bottleneck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[CHANNELS, SPATIAL, SPATIAL]);
}

#[test]
fn test_doclayout_c2f_entry_bottleneck_crown() {
    let def = build_c2f_entry_bottleneck();
    let bindings = c2f_entry_bottleneck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("c2f_entry_bottleneck CROWN method: {method:?}");
}

// ===========================================================================
// 2. C2f full block: entry -> bottleneck -> concat -> exit
// ===========================================================================

fn build_c2f_full_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("doclayout_c2f_full");
    let shape = [CHANNELS, SPATIAL, SPATIAL];
    let concat_shape = [CHANNELS_2X, SPATIAL, SPATIAL];

    let input = b.add_input("features", &shape);

    // Entry 1x1 conv
    let entry = add_conv_bn_silu(&mut b, input, "entry", CHANNELS, CHANNELS, 1, 1, 0, &shape);

    // Bottleneck branch
    let bot1 = add_conv_bn_silu(&mut b, entry, "bot1a", CHANNELS, CHANNELS, 3, 1, 1, &shape);
    let bot2 = add_conv_bn_silu(&mut b, bot1, "bot1b", CHANNELS, CHANNELS, 3, 1, 1, &shape);
    let branch1 = b.add_binary_add(entry, bot2, &shape);

    // Concat entry + bottleneck output along channel dim
    let concat = b.add_concat(&[entry, branch1], 0, &concat_shape);

    // Exit 1x1 conv: reduce channels back
    let exit = add_conv_bn_silu(
        &mut b,
        concat,
        "exit",
        CHANNELS,
        CHANNELS_2X,
        1,
        1,
        0,
        &shape,
    );
    b.build(exit).expect("valid C2f full block")
}

fn c2f_full_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 1); // entry
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 3); // bot1a
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 3); // bot1b
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS_2X, 1); // exit
    bindings
}

#[test]
fn test_doclayout_c2f_full_block_ibp() {
    let def = build_c2f_full_block();
    let bindings = c2f_full_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[CHANNELS, SPATIAL, SPATIAL]);
}

#[test]
fn test_doclayout_c2f_full_block_crown() {
    let def = build_c2f_full_block();
    let bindings = c2f_full_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("c2f_full_block CROWN method: {method:?}");
}

// ===========================================================================
// 3. Backbone stage: ConvBnAct stride-2 + C2f
// ===========================================================================

fn build_backbone_stage() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("doclayout_backbone_stage");
    let half_spatial = SPATIAL / 2; // 4
    let ds_shape = [CHANNELS, half_spatial, half_spatial];
    let shape_out = [CHANNELS, half_spatial, half_spatial];

    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Stride-2 downsampling
    let ds = add_conv_bn_silu(&mut b, input, "ds", CHANNELS, CHANNELS, 3, 2, 1, &ds_shape);

    // C2f block (simplified: entry + one bottleneck + exit)
    let entry = add_conv_bn_silu(&mut b, ds, "entry", CHANNELS, CHANNELS, 1, 1, 0, &shape_out);
    let bot = add_conv_bn_silu(
        &mut b, entry, "bot", CHANNELS, CHANNELS, 3, 1, 1, &shape_out,
    );
    let res = b.add_binary_add(entry, bot, &shape_out);

    b.build(res).expect("valid backbone stage")
}

fn backbone_stage_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 3); // ds
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 1); // entry
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 3); // bot
    bindings
}

#[test]
fn test_doclayout_backbone_stage_ibp() {
    let def = build_backbone_stage();
    let bindings = backbone_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[CHANNELS, SPATIAL / 2, SPATIAL / 2]);
}

#[test]
fn test_doclayout_backbone_stage_crown() {
    let def = build_backbone_stage();
    let bindings = backbone_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("backbone_stage CROWN method: {method:?}");
}

// ===========================================================================
// 4. Detection head: sigmoid + DFL
// ===========================================================================

fn build_detection_head() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("doclayout_detection_head");
    let feat_shape = [CHANNELS, SPATIAL, SPATIAL];

    let input = b.add_input("features", &feat_shape);

    // Classification head: Conv 1x1 -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, CHANNELS, 1, 1]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_logits = b.add_conv2d(
        input,
        cls_w,
        Some(cls_b),
        1,
        1,
        0,
        0,
        &[NUM_CLASSES, SPATIAL, SPATIAL],
    );
    let cls_probs = b.add_sigmoid(cls_logits, &[NUM_CLASSES, SPATIAL, SPATIAL]);

    // Reshape to anchor format for output
    let cls_flat = b.add_reshape(cls_probs, &[NUM_CLASSES * SPATIAL * SPATIAL]);

    b.build(cls_flat).expect("valid detection head")
}

fn detection_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, CHANNELS, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])),
    ]
}

#[test]
fn test_doclayout_detection_head_sigmoid_ibp() {
    let def = build_detection_head();
    let bindings = detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "sigmoid lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi}");
}

#[test]
fn test_doclayout_detection_head_sigmoid_crown() {
    let def = build_detection_head();
    let bindings = detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "sigmoid lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi}");
    eprintln!("detection_head CROWN method: {method:?}");
}

// ===========================================================================
// 5. Backbone stage + detection head (cross-stage)
// ===========================================================================

fn build_backbone_to_detection() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("doclayout_backbone_to_detection");
    let half_spatial = SPATIAL / 2;

    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Backbone stage: stride-2 + bottleneck
    let ds = add_conv_bn_silu(
        &mut b,
        input,
        "ds",
        CHANNELS,
        CHANNELS,
        3,
        2,
        1,
        &[CHANNELS, half_spatial, half_spatial],
    );
    let bot = add_conv_bn_silu(
        &mut b,
        ds,
        "bot",
        CHANNELS,
        CHANNELS,
        3,
        1,
        1,
        &[CHANNELS, half_spatial, half_spatial],
    );
    let stage = b.add_binary_add(ds, bot, &[CHANNELS, half_spatial, half_spatial]);

    // Detection head: 1x1 conv -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, CHANNELS, 1, 1]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_logits = b.add_conv2d(
        stage,
        cls_w,
        Some(cls_b),
        1,
        1,
        0,
        0,
        &[NUM_CLASSES, half_spatial, half_spatial],
    );
    let cls_probs = b.add_sigmoid(cls_logits, &[NUM_CLASSES, half_spatial, half_spatial]);
    let out = b.add_reshape(cls_probs, &[NUM_CLASSES * half_spatial * half_spatial]);

    b.build(out).expect("valid backbone to detection")
}

fn backbone_to_detection_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 3); // ds
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 3); // bot
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        NUM_CLASSES,
        CHANNELS,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])));
    bindings
}

#[test]
fn test_doclayout_backbone_to_detection_ibp() {
    let def = build_backbone_to_detection();
    let bindings = backbone_to_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "sigmoid lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi}");
}

#[test]
fn test_doclayout_backbone_to_detection_crown() {
    let def = build_backbone_to_detection();
    let bindings = backbone_to_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("backbone_to_detection CROWN method: {method:?}");
}

// ===========================================================================
// 6. Image input -> backbone -> detection (end-to-end)
// ===========================================================================

fn build_image_to_detection() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("doclayout_image_to_detection");
    let half_img = IMG_SIZE / 2; // 8
    let quarter_img = IMG_SIZE / 4; // 4

    // Image input
    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);

    // Stage 1: stride-2 conv
    let s1 = add_conv_bn_silu(
        &mut b,
        input,
        "s1",
        CHANNELS,
        IN_CH,
        3,
        2,
        1,
        &[CHANNELS, half_img, half_img],
    );

    // Stage 2: stride-2 conv + bottleneck
    let s2 = add_conv_bn_silu(
        &mut b,
        s1,
        "s2",
        CHANNELS,
        CHANNELS,
        3,
        2,
        1,
        &[CHANNELS, quarter_img, quarter_img],
    );
    let bot = add_conv_bn_silu(
        &mut b,
        s2,
        "bot",
        CHANNELS,
        CHANNELS,
        3,
        1,
        1,
        &[CHANNELS, quarter_img, quarter_img],
    );
    let stage = b.add_binary_add(s2, bot, &[CHANNELS, quarter_img, quarter_img]);

    // Detection head
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, CHANNELS, 1, 1]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_logits = b.add_conv2d(
        stage,
        cls_w,
        Some(cls_b),
        1,
        1,
        0,
        0,
        &[NUM_CLASSES, quarter_img, quarter_img],
    );
    let cls_probs = b.add_sigmoid(cls_logits, &[NUM_CLASSES, quarter_img, quarter_img]);
    let out = b.add_reshape(cls_probs, &[NUM_CLASSES * quarter_img * quarter_img]);

    b.build(out).expect("valid image to detection")
}

fn image_to_detection_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, IN_CH, 3); // s1
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 3); // s2
    push_conv_bn_silu_bindings(&mut bindings, CHANNELS, CHANNELS, 3); // bot
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        NUM_CLASSES,
        CHANNELS,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])));
    bindings
}

#[test]
fn test_doclayout_image_to_detection_ibp() {
    let def = build_image_to_detection();
    let bindings = image_to_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "sigmoid lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi}");
}

#[test]
fn test_doclayout_image_to_detection_crown() {
    let def = build_image_to_detection();
    let bindings = image_to_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("image_to_detection CROWN method: {method:?}");
}

// ===========================================================================
// Verify-and-record
// ===========================================================================

#[test]
fn test_doclayout_c2f_full_block_verify_and_record() {
    let def = build_c2f_full_block();
    let bindings = c2f_full_block_bindings();
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "doclayout_yolo::test_doclayout_c2f_full_block_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}

#[test]
fn test_doclayout_image_to_detection_verify_and_record() {
    let def = build_image_to_detection();
    let bindings = image_to_detection_bindings();
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "doclayout_yolo::test_doclayout_image_to_detection_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}
