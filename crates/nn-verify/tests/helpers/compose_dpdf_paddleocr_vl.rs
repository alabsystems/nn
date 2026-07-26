// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for PaddleOCR-VL text detection subpipeline bounds.
//!
//! Verifies IBP and CROWN bound propagation through the PaddleOCR-VL text
//! detection pipeline used in the dpdf document understanding system.
//! PaddleOCR-VL extends the standard PaddleOCR DB detector with a
//! vision-language backbone for improved scene text detection.
//!
//! ## Tests (15 tests)
//!
//! **Backbone feature extraction (tests 1-2):**
//! 1.  **ResNet backbone Conv-BN-ReLU block** — Conv2d -> BatchNorm -> ReLU (IBP)
//! 2.  **MobileNet backbone depthwise separable block** — DW Conv -> BN -> PW Conv (IBP + CROWN)
//!
//! **FPN neck (tests 3-4):**
//! 3.  **FPN lateral + top-down fusion** — 1x1 conv + add from higher level (IBP)
//! 4.  **FPN multi-scale 3-level** — P3/P4/P5 feature fusion pipeline (IBP + CROWN)
//!
//! **DB text detection head (tests 5-6):**
//! 5.  **DB probability map sigmoid** — Conv -> sigmoid output in [0, 1] (IBP + CROWN)
//! 6.  **DB head with BatchNorm** — Conv -> BN -> ReLU -> Conv -> sigmoid (IBP)
//!
//! **Threshold map (tests 7-8):**
//! 7.  **Adaptive threshold map** — Conv -> sigmoid threshold in [0, 1] (IBP)
//! 8.  **Threshold map CROWN** — tighter bounds via CROWN linearization (CROWN)
//!
//! **Binary map (tests 9-10):**
//! 9.  **DB binary map** — sigmoid(k * (P - T)) approximation in [0, 1] (IBP)
//! 10. **Binary map CROWN** — CROWN linearization of binarization (CROWN)
//!
//! **Post-processing bounds (tests 11-12):**
//! 11. **Polygon extraction bounds** — Binary map -> linear -> sigmoid region confidence (IBP)
//! 12. **Box coordinate regression** — Linear -> sigmoid normalized box coords in [0, 1] (IBP)
//!
//! **Multi-scale + full pipeline (tests 13-15):**
//! 13. **Multi-scale input resolution** — Two resolutions produce bounded outputs (IBP)
//! 14. **Full pipeline composition** — Backbone -> FPN -> DB head -> binary map (IBP)
//! 15. **Full pipeline monotone tightening** — Narrower input -> narrower output (IBP)
//!
//! Architecture references:
//! - PaddleOCR (Baidu): Production OCR with DB detector + SVTR recognizer
//! - PaddleOCR-VL: Vision-language enhanced PaddleOCR
//! - DB (Liao et al. 2020): Differentiable Binarization for text detection
//! - MobileNetV3 (Howard et al. 2019): Lightweight backbone for mobile OCR
//! - ResNet (He et al. 2016): Standard backbone for text detection
//! - FPN (Lin et al. 2017): Feature Pyramid Network for multi-scale fusion
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=16, BACKBONE_CH=8, FPN_CH=16, MID_CH=16
//! - P3 spatial=4, P4 spatial=2, P5 spatial=1
//!
//! Part of #4222: NY compose tests for PaddleOCR-VL text detection.

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

/// Input image spatial size.
const IMG_SIZE: usize = 16;
/// Input channels (RGB).
const IN_CH: usize = 3;
/// Backbone output channels.
const BACKBONE_CH: usize = 8;
/// FPN unified channel width.
const FPN_CH: usize = 16;
/// Intermediate channels.
const MID_CH: usize = 16;
/// Single-channel map output (probability/threshold/binary).
const MAP_CH: usize = 1;
/// P3 spatial size (IMG_SIZE / 4).
const P3_SPATIAL: usize = 4;
/// P4 spatial size (IMG_SIZE / 8).
const P4_SPATIAL: usize = 2;
/// P5 spatial size (IMG_SIZE / 16).
const P5_SPATIAL: usize = 1;
/// Weight magnitude for bounded verification.
const W_MAG: f32 = 0.02;
/// DB differentiable binarization expansion factor k.
const DB_K: f32 = 50.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor of given shape.
fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), W_MAG)
}

/// Ones tensor (for BN weight / variance).
fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

/// Zeros tensor (for BN mean / bias).
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

/// Push Conv-BN bindings (7 params: conv_w, conv_b, bn_mean, bn_var, bn_w, bn_b, eps).
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

/// Add Conv-BN-ReLU block to builder.
///
/// Returns output node ID after ReLU. Adds 7 input nodes
/// (conv_w, conv_b, bn_mean, bn_var, bn_w, bn_b, eps).
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
// 1. ResNet backbone Conv-BN-ReLU block (IBP)
// ===========================================================================

/// Build a ResNet-style backbone block: Conv2d -> BatchNorm -> ReLU.
///
/// Input: [IN_CH, IMG_SIZE, IMG_SIZE] (RGB image in [0, 1])
/// Output: [BACKBONE_CH, IMG_SIZE, IMG_SIZE] (feature map)
fn build_resnet_backbone_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddleocr_vl_resnet_backbone");
    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);
    let out = add_conv_bn_relu(
        &mut b,
        input,
        "res",
        IN_CH,
        BACKBONE_CH,
        3,
        1,
        1,
        IMG_SIZE,
        IMG_SIZE,
    );
    b.build(out).expect("valid ResNet backbone block")
}

fn resnet_backbone_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_bindings(&mut bindings, BACKBONE_CH, IN_CH, 3);
    bindings
}

#[test]
fn test_paddleocr_vl_resnet_backbone_ibp() {
    let def = build_resnet_backbone_block();
    let bindings = resnet_backbone_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ResNet backbone");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE],
        "backbone output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ResNet backbone IBP: bounds=[{lo_min}, {hi_max}]");
    // After ReLU, lower bound >= 0
    assert!(lo_min >= -1e-6, "ReLU lower >= 0, got {lo_min}");
}

// ===========================================================================
// 2. MobileNet backbone depthwise separable block (IBP + CROWN)
// ===========================================================================

/// Build a MobileNet-style depthwise separable conv block.
///
/// Depthwise: Conv2d(groups=in_ch) -> BN -> ReLU
/// Pointwise: Conv2d(1x1) -> BN -> ReLU
///
/// Since grouped convolutions aren't directly supported in the builder,
/// we approximate with: Conv2d(3x3, in=in_ch, out=in_ch) -> BN -> ReLU
/// -> Conv2d(1x1, in=in_ch, out=out_ch) -> BN -> ReLU.
///
/// Input: [IN_CH, IMG_SIZE, IMG_SIZE]
/// Output: [BACKBONE_CH, IMG_SIZE, IMG_SIZE]
fn build_mobilenet_dw_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddleocr_vl_mobilenet_dw");
    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);

    // Depthwise-like 3x3 conv (same channels in/out)
    let dw_out = add_conv_bn_relu(
        &mut b, input, "dw", IN_CH, IN_CH, 3, 1, 1, IMG_SIZE, IMG_SIZE,
    );

    // Pointwise 1x1 conv for channel expansion
    let pw_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let pw_w = b.add_input("pw_conv_w", &[BACKBONE_CH, IN_CH, 1, 1]);
    let pw_b = b.add_input("pw_conv_b", &[BACKBONE_CH]);
    let pw_conv = b.add_conv2d(dw_out, pw_w, Some(pw_b), 1, 1, 0, 0, &pw_shape);

    let bn_mean = b.add_input("pw_bn_mean", &[BACKBONE_CH]);
    let bn_var = b.add_input("pw_bn_var", &[BACKBONE_CH]);
    let bn_w = b.add_input("pw_bn_w", &[BACKBONE_CH]);
    let bn_b = b.add_input("pw_bn_b", &[BACKBONE_CH]);
    let eps = b.add_input("pw_eps", &[1]);
    let bn = b.add_batch_norm(pw_conv, bn_mean, bn_var, bn_w, bn_b, eps, &pw_shape);
    let out = b.add_relu(bn, &pw_shape);

    b.build(out).expect("valid MobileNet DW block")
}

fn mobilenet_dw_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    // Depthwise conv-bn (IN_CH -> IN_CH, 3x3)
    push_conv_bn_bindings(&mut bindings, IN_CH, IN_CH, 3);
    // Pointwise conv (IN_CH -> BACKBONE_CH, 1x1) + BN
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        BACKBONE_CH,
        IN_CH,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[BACKBONE_CH])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[BACKBONE_CH])));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[BACKBONE_CH])));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[BACKBONE_CH])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[BACKBONE_CH])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings
}

#[test]
fn test_paddleocr_vl_mobilenet_dw_ibp() {
    let def = build_mobilenet_dw_block();
    let bindings = mobilenet_dw_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MobileNet DW block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE],
        "MobileNet DW output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MobileNet DW IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "ReLU lower >= 0, got {lo_min}");
}

#[test]
fn test_paddleocr_vl_mobilenet_dw_crown() {
    let def = build_mobilenet_dw_block();
    let bindings = mobilenet_dw_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MobileNet DW CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min >= -1e-6, "ReLU lower >= 0, got {lo_min}");
}

// ===========================================================================
// 3. FPN lateral + top-down fusion (IBP)
// ===========================================================================

/// Build FPN lateral connection + top-down addition.
///
/// Takes P5-level features (coarse, small spatial) and fuses with P4
/// (finer, larger spatial) via 1x1 conv lateral + elementwise addition.
///
/// Input: [BACKBONE_CH, P3_SPATIAL, P3_SPATIAL] (P4-level features)
/// Lateral: 1x1 Conv2d channel adjustment
/// Addition: P5 features (via 1x1 conv) upsampled and added.
///
/// Output: [FPN_CH, P3_SPATIAL, P3_SPATIAL] (fused features).
fn build_fpn_lateral_fusion() -> TensorKernelDef {
    let in_shape = [BACKBONE_CH, P3_SPATIAL, P3_SPATIAL];
    let out_shape = [FPN_CH, P3_SPATIAL, P3_SPATIAL];

    let mut b = TensorBlockBuilder::new("paddleocr_vl_fpn_lateral");

    // P4-level input
    let input = b.add_input("p4_features", &in_shape);

    // Lateral 1x1 conv: [BACKBONE_CH, 4, 4] -> [FPN_CH, 4, 4]
    let lat_w = b.add_input("lateral_w", &[FPN_CH, BACKBONE_CH, 1, 1]);
    let lat_b = b.add_input("lateral_b", &[FPN_CH]);
    let lateral = b.add_conv2d(input, lat_w, Some(lat_b), 1, 1, 0, 0, &out_shape);

    // Top-down features (simulated as separate 1x1 conv from input)
    // In practice this would come from upsampled P5; we model the fusion point.
    let td_w = b.add_input("topdown_w", &[FPN_CH, BACKBONE_CH, 1, 1]);
    let td_b = b.add_input("topdown_b", &[FPN_CH]);
    let topdown = b.add_conv2d(input, td_w, Some(td_b), 1, 1, 0, 0, &out_shape);

    // Fuse: lateral + top-down
    let out = b.add_binary_add(lateral, topdown, &out_shape);

    b.build(out).expect("valid FPN lateral fusion")
}

fn fpn_lateral_fusion_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // Lateral conv
        TensorParamBinding::ConstantTensor(w(&[FPN_CH, BACKBONE_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])),
        // Top-down conv
        TensorParamBinding::ConstantTensor(w(&[FPN_CH, BACKBONE_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])),
    ]
}

#[test]
fn test_paddleocr_vl_fpn_lateral_fusion_ibp() {
    let def = build_fpn_lateral_fusion();
    let bindings = fpn_lateral_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[BACKBONE_CH, P3_SPATIAL, P3_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FPN lateral fusion");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[FPN_CH, P3_SPATIAL, P3_SPATIAL],
        "FPN lateral fusion output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FPN lateral fusion IBP: bounds=[{lo_min}, {hi_max}]");
}

// ===========================================================================
// 4. FPN multi-scale 3-level (IBP + CROWN)
// ===========================================================================

/// Build 3-level FPN: three parallel 1x1 convs + ReLU on P3-level features.
///
/// Models the channel alignment step of an FPN where backbone features
/// at three levels (here all derived from one input for simplicity) are
/// projected to a unified channel dimension via 1x1 conv.
///
/// Input: [BACKBONE_CH, P3_SPATIAL, P3_SPATIAL]
/// Three branches: 1x1 conv(BACKBONE_CH -> FPN_CH) -> ReLU each
/// Concat on channel axis -> [FPN_CH*3, P3_SPATIAL, P3_SPATIAL]
/// Final 1x1 conv -> [FPN_CH, P3_SPATIAL, P3_SPATIAL]
fn build_fpn_multiscale_3level() -> TensorKernelDef {
    let in_shape = [BACKBONE_CH, P3_SPATIAL, P3_SPATIAL];
    let branch_shape = [FPN_CH, P3_SPATIAL, P3_SPATIAL];
    let concat_ch = FPN_CH * 3;
    let concat_shape = [concat_ch, P3_SPATIAL, P3_SPATIAL];
    let out_shape = [FPN_CH, P3_SPATIAL, P3_SPATIAL];

    let mut b = TensorBlockBuilder::new("paddleocr_vl_fpn_3level");
    let input = b.add_input("features", &in_shape);

    // Branch 1 (P3-level): 1x1 conv + ReLU
    let w1 = b.add_input("p3_w", &[FPN_CH, BACKBONE_CH, 1, 1]);
    let b1 = b.add_input("p3_b", &[FPN_CH]);
    let p3 = b.add_conv2d(input, w1, Some(b1), 1, 1, 0, 0, &branch_shape);
    let p3_relu = b.add_relu(p3, &branch_shape);

    // Branch 2 (P4-level): 1x1 conv + ReLU
    let w2 = b.add_input("p4_w", &[FPN_CH, BACKBONE_CH, 1, 1]);
    let b2 = b.add_input("p4_b", &[FPN_CH]);
    let p4 = b.add_conv2d(input, w2, Some(b2), 1, 1, 0, 0, &branch_shape);
    let p4_relu = b.add_relu(p4, &branch_shape);

    // Branch 3 (P5-level): 1x1 conv + ReLU
    let w3 = b.add_input("p5_w", &[FPN_CH, BACKBONE_CH, 1, 1]);
    let b3 = b.add_input("p5_b", &[FPN_CH]);
    let p5 = b.add_conv2d(input, w3, Some(b3), 1, 1, 0, 0, &branch_shape);
    let p5_relu = b.add_relu(p5, &branch_shape);

    // Concat all levels
    let fused = b.add_concat(&[p3_relu, p4_relu, p5_relu], 0, &concat_shape);

    // Merge 1x1 conv: [FPN_CH*3] -> [FPN_CH]
    let wm = b.add_input("merge_w", &[FPN_CH, concat_ch, 1, 1]);
    let bm = b.add_input("merge_b", &[FPN_CH]);
    let out = b.add_conv2d(fused, wm, Some(bm), 1, 1, 0, 0, &out_shape);

    b.build(out).expect("valid FPN 3-level")
}

fn fpn_multiscale_3level_bindings() -> Vec<TensorParamBinding> {
    let concat_ch = FPN_CH * 3;
    vec![
        TensorParamBinding::Variable,
        // P3 branch
        TensorParamBinding::ConstantTensor(w(&[FPN_CH, BACKBONE_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])),
        // P4 branch
        TensorParamBinding::ConstantTensor(w(&[FPN_CH, BACKBONE_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])),
        // P5 branch
        TensorParamBinding::ConstantTensor(w(&[FPN_CH, BACKBONE_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])),
        // Merge conv
        TensorParamBinding::ConstantTensor(w(&[FPN_CH, concat_ch, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])),
    ]
}

#[test]
fn test_paddleocr_vl_fpn_3level_ibp() {
    let def = build_fpn_multiscale_3level();
    let bindings = fpn_multiscale_3level_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[BACKBONE_CH, P3_SPATIAL, P3_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FPN 3-level");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[FPN_CH, P3_SPATIAL, P3_SPATIAL],
        "FPN 3-level output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FPN 3-level IBP: bounds=[{lo_min}, {hi_max}]");
}

#[test]
fn test_paddleocr_vl_fpn_3level_crown() {
    let def = build_fpn_multiscale_3level();
    let bindings = fpn_multiscale_3level_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[BACKBONE_CH, P3_SPATIAL, P3_SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[FPN_CH, P3_SPATIAL, P3_SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FPN 3-level CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 5. DB probability map sigmoid (IBP + CROWN)
// ===========================================================================

/// Build DB probability map head: Conv -> ReLU -> Conv -> sigmoid.
///
/// Input: [FPN_CH, P3_SPATIAL, P3_SPATIAL] (FPN features)
/// Output: [MAP_CH, P3_SPATIAL, P3_SPATIAL] (probability map in (0, 1))
fn build_db_prob_map() -> TensorKernelDef {
    let mid_shape = [MID_CH, P3_SPATIAL, P3_SPATIAL];
    let out_shape = [MAP_CH, P3_SPATIAL, P3_SPATIAL];

    let mut b = TensorBlockBuilder::new("paddleocr_vl_db_prob_map");
    let input = b.add_input("fpn_features", &[FPN_CH, P3_SPATIAL, P3_SPATIAL]);

    // Conv2d #1: [FPN_CH] -> [MID_CH]
    let w1 = b.add_input("prob_conv1_w", &[MID_CH, FPN_CH, 3, 3]);
    let b1 = b.add_input("prob_conv1_b", &[MID_CH]);
    let conv1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 1, 1, &mid_shape);
    let relu = b.add_relu(conv1, &mid_shape);

    // Conv2d #2: [MID_CH] -> [MAP_CH]
    let w2 = b.add_input("prob_conv2_w", &[MAP_CH, MID_CH, 1, 1]);
    let b2 = b.add_input("prob_conv2_b", &[MAP_CH]);
    let conv2 = b.add_conv2d(relu, w2, Some(b2), 1, 1, 0, 0, &out_shape);

    // Sigmoid: output in (0, 1)
    let out = b.add_sigmoid(conv2, &out_shape);

    b.build(out).expect("valid DB prob map")
}

fn db_prob_map_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[MID_CH, FPN_CH, 3, 3])),
        TensorParamBinding::ConstantTensor(zeros(&[MID_CH])),
        TensorParamBinding::ConstantTensor(w(&[MAP_CH, MID_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[MAP_CH])),
    ]
}

#[test]
fn test_paddleocr_vl_db_prob_map_ibp() {
    let def = build_db_prob_map();
    let bindings = db_prob_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FPN_CH, P3_SPATIAL, P3_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DB prob map");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, P3_SPATIAL, P3_SPATIAL],
        "DB prob map output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DB prob map IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

#[test]
fn test_paddleocr_vl_db_prob_map_crown() {
    let def = build_db_prob_map();
    let bindings = db_prob_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FPN_CH, P3_SPATIAL, P3_SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, P3_SPATIAL, P3_SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DB prob map CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 6. DB head with BatchNorm (IBP)
// ===========================================================================

/// Build DB detection head with BatchNorm: Conv -> BN -> ReLU -> Conv -> sigmoid.
///
/// Input: [BACKBONE_CH, P3_SPATIAL, P3_SPATIAL]
/// Output: [MAP_CH, P3_SPATIAL, P3_SPATIAL] (probability map in (0, 1))
fn build_db_head_with_bn() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddleocr_vl_db_head_bn");
    let input = b.add_input("features", &[BACKBONE_CH, P3_SPATIAL, P3_SPATIAL]);

    // Conv-BN-ReLU
    let mid = add_conv_bn_relu(
        &mut b,
        input,
        "head",
        BACKBONE_CH,
        MID_CH,
        3,
        1,
        1,
        P3_SPATIAL,
        P3_SPATIAL,
    );

    // Final Conv -> sigmoid
    let out_shape = [MAP_CH, P3_SPATIAL, P3_SPATIAL];
    let w2 = b.add_input("head_final_w", &[MAP_CH, MID_CH, 1, 1]);
    let b2 = b.add_input("head_final_b", &[MAP_CH]);
    let conv2 = b.add_conv2d(mid, w2, Some(b2), 1, 1, 0, 0, &out_shape);
    let out = b.add_sigmoid(conv2, &out_shape);

    b.build(out).expect("valid DB head with BN")
}

fn db_head_with_bn_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_bindings(&mut bindings, MID_CH, BACKBONE_CH, 3);
    // Final conv
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        MAP_CH, MID_CH, 1, 1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[MAP_CH])));
    bindings
}

#[test]
fn test_paddleocr_vl_db_head_bn_ibp() {
    let def = build_db_head_with_bn();
    let bindings = db_head_with_bn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[BACKBONE_CH, P3_SPATIAL, P3_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DB head with BN");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, P3_SPATIAL, P3_SPATIAL],
        "DB head BN output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DB head BN IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 7. Adaptive threshold map (IBP)
// ===========================================================================

/// Build adaptive threshold map: Conv2d -> sigmoid.
///
/// The DB detector has a separate threshold map head that predicts
/// per-pixel adaptive thresholds in (0, 1) for binarization.
///
/// Input: [FPN_CH, P3_SPATIAL, P3_SPATIAL]
/// Output: [MAP_CH, P3_SPATIAL, P3_SPATIAL] (threshold in (0, 1))
fn build_threshold_map() -> TensorKernelDef {
    let out_shape = [MAP_CH, P3_SPATIAL, P3_SPATIAL];
    let mut b = TensorBlockBuilder::new("paddleocr_vl_threshold_map");

    let input = b.add_input("features", &[FPN_CH, P3_SPATIAL, P3_SPATIAL]);
    let tw = b.add_input("thresh_w", &[MAP_CH, FPN_CH, 3, 3]);
    let tb = b.add_input("thresh_b", &[MAP_CH]);
    let conv = b.add_conv2d(input, tw, Some(tb), 1, 1, 1, 1, &out_shape);
    let out = b.add_sigmoid(conv, &out_shape);

    b.build(out).expect("valid threshold map")
}

fn threshold_map_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[MAP_CH, FPN_CH, 3, 3])),
        TensorParamBinding::ConstantTensor(zeros(&[MAP_CH])),
    ]
}

#[test]
fn test_paddleocr_vl_threshold_map_ibp() {
    let def = build_threshold_map();
    let bindings = threshold_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FPN_CH, P3_SPATIAL, P3_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through threshold map");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, P3_SPATIAL, P3_SPATIAL],
        "threshold map output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Threshold map IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= 0.0 - 1e-6, "threshold lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "threshold upper <= 1, got {hi_max}");
}

// ===========================================================================
// 8. Threshold map CROWN
// ===========================================================================

#[test]
fn test_paddleocr_vl_threshold_map_crown() {
    let def = build_threshold_map();
    let bindings = threshold_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FPN_CH, P3_SPATIAL, P3_SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, P3_SPATIAL, P3_SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Threshold map CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min >= 0.0 - 1e-6, "threshold lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "threshold upper <= 1, got {hi_max}");
}

// ===========================================================================
// 9. DB binary map: sigmoid(k * (P - T)) (IBP)
// ===========================================================================

/// Build DB binary map: sigmoid(k * (P - T)).
///
/// The binarization approximation: B = sigmoid(k * (P - T)), where
/// k is a large expansion factor (~50), P is the probability map,
/// and T is the threshold map. Both P and T are in (0, 1).
///
/// Modeled as: 2-channel input [P; T], 1x1 Conv with weights [+k, -k],
/// followed by sigmoid.
///
/// Input: [2, P3_SPATIAL, P3_SPATIAL] (P and T stacked)
/// Output: [MAP_CH, P3_SPATIAL, P3_SPATIAL] (binary map in (0, 1))
fn build_binary_map() -> TensorKernelDef {
    let in_shape = [2, P3_SPATIAL, P3_SPATIAL];
    let out_shape = [MAP_CH, P3_SPATIAL, P3_SPATIAL];

    let mut b = TensorBlockBuilder::new("paddleocr_vl_binary_map");
    let input = b.add_input("prob_thresh", &in_shape);

    // 1x1 Conv2d: weight[0,0]=+k, weight[0,1]=-k computes k*(P-T)
    let bw = b.add_input("binary_w", &[MAP_CH, 2, 1, 1]);
    let bb = b.add_input("binary_b", &[MAP_CH]);
    let diff = b.add_conv2d(input, bw, Some(bb), 1, 1, 0, 0, &out_shape);
    let out = b.add_sigmoid(diff, &out_shape);

    b.build(out).expect("valid binary map")
}

fn binary_map_bindings() -> Vec<TensorParamBinding> {
    // Weight encodes k*(P - T): channel 0 (P) gets +k, channel 1 (T) gets -k
    let mut w_data = vec![0.0f32; MAP_CH * 2 * 1 * 1];
    w_data[0] = DB_K; // weight for P channel
    w_data[1] = -DB_K; // weight for T channel
    let bw = ArrayD::from_shape_vec(IxDyn(&[MAP_CH, 2, 1, 1]), w_data).expect("valid weight");

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(bw),
        TensorParamBinding::ConstantTensor(zeros(&[MAP_CH])),
    ]
}

#[test]
fn test_paddleocr_vl_binary_map_ibp() {
    let def = build_binary_map();
    let bindings = binary_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // P and T both in [0.2, 0.8] (typical after sigmoid)
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, P3_SPATIAL, P3_SPATIAL]), 0.2f32),
        ArrayD::from_elem(IxDyn(&[2, P3_SPATIAL, P3_SPATIAL]), 0.8f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through binary map");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, P3_SPATIAL, P3_SPATIAL],
        "binary map output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Binary map IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= 0.0 - 1e-6, "binary map lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "binary map upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. Binary map CROWN
// ===========================================================================

#[test]
fn test_paddleocr_vl_binary_map_crown() {
    let def = build_binary_map();
    let bindings = binary_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Narrower bounds for CROWN precision
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, P3_SPATIAL, P3_SPATIAL]), 0.3f32),
        ArrayD::from_elem(IxDyn(&[2, P3_SPATIAL, P3_SPATIAL]), 0.7f32),
    )
    .expect("valid bounds");

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, P3_SPATIAL, P3_SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Binary map CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min >= 0.0 - 1e-6, "binary map lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "binary map upper <= 1, got {hi_max}");
}

// ===========================================================================
// 11. Polygon extraction bounds: binary map -> linear -> sigmoid confidence (IBP)
// ===========================================================================

/// Build post-processing confidence head: Conv2d -> sigmoid.
///
/// After the binary map, text regions are extracted. This models the
/// confidence scoring of extracted polygonal regions via a small conv head.
///
/// Input: [MAP_CH, P3_SPATIAL, P3_SPATIAL] (binary map features)
/// Output: [MAP_CH, P3_SPATIAL, P3_SPATIAL] (region confidence in (0, 1))
fn build_polygon_confidence() -> TensorKernelDef {
    let in_shape = [MAP_CH, P3_SPATIAL, P3_SPATIAL];
    let mid_shape = [BACKBONE_CH, P3_SPATIAL, P3_SPATIAL];
    let out_shape = [MAP_CH, P3_SPATIAL, P3_SPATIAL];

    let mut b = TensorBlockBuilder::new("paddleocr_vl_polygon_conf");
    let input = b.add_input("binary_map", &in_shape);

    // Expand channels
    let w1 = b.add_input("conf_w1", &[BACKBONE_CH, MAP_CH, 3, 3]);
    let b1 = b.add_input("conf_b1", &[BACKBONE_CH]);
    let conv1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 1, 1, &mid_shape);
    let relu = b.add_relu(conv1, &mid_shape);

    // Project to confidence score
    let w2 = b.add_input("conf_w2", &[MAP_CH, BACKBONE_CH, 1, 1]);
    let b2 = b.add_input("conf_b2", &[MAP_CH]);
    let conv2 = b.add_conv2d(relu, w2, Some(b2), 1, 1, 0, 0, &out_shape);
    let out = b.add_sigmoid(conv2, &out_shape);

    b.build(out).expect("valid polygon confidence")
}

fn polygon_confidence_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[BACKBONE_CH, MAP_CH, 3, 3])),
        TensorParamBinding::ConstantTensor(zeros(&[BACKBONE_CH])),
        TensorParamBinding::ConstantTensor(w(&[MAP_CH, BACKBONE_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[MAP_CH])),
    ]
}

#[test]
fn test_paddleocr_vl_polygon_confidence_ibp() {
    let def = build_polygon_confidence();
    let bindings = polygon_confidence_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Binary map input in [0, 1]
    let input = image_bounds(&[MAP_CH, P3_SPATIAL, P3_SPATIAL]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through polygon confidence");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, P3_SPATIAL, P3_SPATIAL],
        "polygon confidence output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Polygon confidence IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= 0.0 - 1e-6, "confidence lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "confidence upper <= 1, got {hi_max}");
}

// ===========================================================================
// 12. Box coordinate regression: Linear -> sigmoid (IBP)
// ===========================================================================

/// Build box coordinate regression head: Conv -> sigmoid for normalized coords.
///
/// Models the bounding box regression head that outputs normalized
/// coordinates in [0, 1] via sigmoid.
///
/// Input: [FPN_CH, P3_SPATIAL, P3_SPATIAL]
/// Output: [4, P3_SPATIAL, P3_SPATIAL] (x, y, w, h normalized coords)
fn build_box_regression() -> TensorKernelDef {
    let box_ch = 4usize; // x, y, w, h
    let out_shape = [box_ch, P3_SPATIAL, P3_SPATIAL];

    let mut b = TensorBlockBuilder::new("paddleocr_vl_box_regression");
    let input = b.add_input("features", &[FPN_CH, P3_SPATIAL, P3_SPATIAL]);

    let bw = b.add_input("box_w", &[box_ch, FPN_CH, 1, 1]);
    let bb = b.add_input("box_b", &[box_ch]);
    let conv = b.add_conv2d(input, bw, Some(bb), 1, 1, 0, 0, &out_shape);
    let out = b.add_sigmoid(conv, &out_shape);

    b.build(out).expect("valid box regression")
}

fn box_regression_bindings() -> Vec<TensorParamBinding> {
    let box_ch = 4usize;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[box_ch, FPN_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[box_ch])),
    ]
}

#[test]
fn test_paddleocr_vl_box_regression_ibp() {
    let def = build_box_regression();
    let bindings = box_regression_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FPN_CH, P3_SPATIAL, P3_SPATIAL], 1.0);

    let box_ch = 4usize;
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through box regression");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[box_ch, P3_SPATIAL, P3_SPATIAL],
        "box regression output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Box regression IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 13. Multi-scale input resolution (IBP)
// ===========================================================================

/// Verify that different input spatial resolutions produce bounded outputs.
///
/// Uses the DB probability map head at two different spatial sizes
/// to verify bounds hold regardless of input resolution.
#[test]
fn test_paddleocr_vl_multiscale_input_ibp() {
    // Resolution 1: P3_SPATIAL x P3_SPATIAL = 4x4
    {
        let def = build_db_prob_map();
        let bindings = db_prob_map_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[FPN_CH, P3_SPATIAL, P3_SPATIAL], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP at resolution 1");

        assert_bounds_valid(&output);
        let (lo_min_1, hi_max_1) = bounds_min_max(&output);
        eprintln!("Multi-scale res1 ({P3_SPATIAL}x{P3_SPATIAL}): bounds=[{lo_min_1}, {hi_max_1}]");
        assert!(lo_min_1 >= 0.0 - 1e-6, "res1 lower >= 0");
        assert!(hi_max_1 <= 1.0 + 1e-6, "res1 upper <= 1");
    }

    // Resolution 2: build a second head at P4 spatial (2x2)
    {
        let mid_shape = [MID_CH, P4_SPATIAL, P4_SPATIAL];
        let out_shape = [MAP_CH, P4_SPATIAL, P4_SPATIAL];

        let mut b = TensorBlockBuilder::new("paddleocr_vl_db_prob_map_p4");
        let input_node = b.add_input("fpn_features", &[FPN_CH, P4_SPATIAL, P4_SPATIAL]);

        let w1 = b.add_input("conv1_w", &[MID_CH, FPN_CH, 3, 3]);
        let b1 = b.add_input("conv1_b", &[MID_CH]);
        let conv1 = b.add_conv2d(input_node, w1, Some(b1), 1, 1, 1, 1, &mid_shape);
        let relu = b.add_relu(conv1, &mid_shape);

        let w2 = b.add_input("conv2_w", &[MAP_CH, MID_CH, 1, 1]);
        let b2 = b.add_input("conv2_b", &[MAP_CH]);
        let conv2 = b.add_conv2d(relu, w2, Some(b2), 1, 1, 0, 0, &out_shape);
        let out = b.add_sigmoid(conv2, &out_shape);

        let def = b.build(out).expect("valid prob map at P4");
        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(w(&[MID_CH, FPN_CH, 3, 3])),
            TensorParamBinding::ConstantTensor(zeros(&[MID_CH])),
            TensorParamBinding::ConstantTensor(w(&[MAP_CH, MID_CH, 1, 1])),
            TensorParamBinding::ConstantTensor(zeros(&[MAP_CH])),
        ];

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[FPN_CH, P4_SPATIAL, P4_SPATIAL], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP at resolution 2");

        assert_bounds_valid(&output);
        let (lo_min_2, hi_max_2) = bounds_min_max(&output);
        eprintln!("Multi-scale res2 ({P4_SPATIAL}x{P4_SPATIAL}): bounds=[{lo_min_2}, {hi_max_2}]");
        assert!(lo_min_2 >= 0.0 - 1e-6, "res2 lower >= 0");
        assert!(hi_max_2 <= 1.0 + 1e-6, "res2 upper <= 1");
    }
}

// ===========================================================================
// 14. Full pipeline composition: Backbone -> FPN -> DB head -> binary map (IBP)
// ===========================================================================

/// Build full PaddleOCR-VL text detection pipeline.
///
/// Input: [IN_CH, IMG_SIZE, IMG_SIZE] (RGB image in [0, 1])
/// Stage 1: Conv-BN-ReLU backbone -> [BACKBONE_CH, IMG_SIZE, IMG_SIZE]
/// Stage 2: 1x1 conv (FPN lateral) -> [FPN_CH, IMG_SIZE, IMG_SIZE]
/// Stage 3: Conv-ReLU-Conv-sigmoid (DB prob head) -> [MAP_CH, IMG_SIZE, IMG_SIZE]
///
/// Output: [MAP_CH, IMG_SIZE, IMG_SIZE] (probability map in (0, 1))
fn build_full_pipeline() -> TensorKernelDef {
    let fpn_shape = [FPN_CH, IMG_SIZE, IMG_SIZE];
    let mid_shape = [MID_CH, IMG_SIZE, IMG_SIZE];
    let out_shape = [MAP_CH, IMG_SIZE, IMG_SIZE];

    let mut b = TensorBlockBuilder::new("paddleocr_vl_full_pipeline");
    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);

    // Stage 1: Backbone Conv-BN-ReLU
    let backbone_out = add_conv_bn_relu(
        &mut b,
        input,
        "bb",
        IN_CH,
        BACKBONE_CH,
        3,
        1,
        1,
        IMG_SIZE,
        IMG_SIZE,
    );

    // Stage 2: FPN lateral 1x1 conv
    let fpn_w = b.add_input("fpn_w", &[FPN_CH, BACKBONE_CH, 1, 1]);
    let fpn_b = b.add_input("fpn_b", &[FPN_CH]);
    let fpn_out = b.add_conv2d(backbone_out, fpn_w, Some(fpn_b), 1, 1, 0, 0, &fpn_shape);

    // Stage 3: DB head — Conv -> ReLU -> Conv -> sigmoid
    let h_w1 = b.add_input("head_w1", &[MID_CH, FPN_CH, 3, 3]);
    let h_b1 = b.add_input("head_b1", &[MID_CH]);
    let h_conv1 = b.add_conv2d(fpn_out, h_w1, Some(h_b1), 1, 1, 1, 1, &mid_shape);
    let h_relu = b.add_relu(h_conv1, &mid_shape);

    let h_w2 = b.add_input("head_w2", &[MAP_CH, MID_CH, 1, 1]);
    let h_b2 = b.add_input("head_b2", &[MAP_CH]);
    let h_conv2 = b.add_conv2d(h_relu, h_w2, Some(h_b2), 1, 1, 0, 0, &out_shape);
    let out = b.add_sigmoid(h_conv2, &out_shape);

    b.build(out).expect("valid full pipeline")
}

fn full_pipeline_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    // Backbone Conv-BN-ReLU
    push_conv_bn_bindings(&mut bindings, BACKBONE_CH, IN_CH, 3);
    // FPN lateral 1x1
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FPN_CH,
        BACKBONE_CH,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])));
    // DB head conv1
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        MID_CH, FPN_CH, 3, 3,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[MID_CH])));
    // DB head conv2
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        MAP_CH, MID_CH, 1, 1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[MAP_CH])));
    bindings
}

#[test]
fn test_paddleocr_vl_full_pipeline_ibp() {
    let def = build_full_pipeline();
    let bindings = full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, IMG_SIZE, IMG_SIZE],
        "full pipeline output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full pipeline IBP: bounds=[{lo_min}, {hi_max}]");
    // End-to-end: sigmoid guarantees (0, 1)
    assert!(lo_min >= 0.0 - 1e-6, "pipeline lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "pipeline upper <= 1, got {hi_max}");
}

// ===========================================================================
// 15. Full pipeline monotone tightening (IBP)
// ===========================================================================

/// Verify that narrower input bounds produce narrower output bounds.
///
/// Tests with epsilon = 1.0 (full [0,1]) and epsilon = 0.5 (center crop),
/// asserts that tighter input yields tighter output.
#[test]
fn test_paddleocr_vl_full_pipeline_monotone_tightening() {
    let def = build_full_pipeline();
    let bindings = full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input: [0, 1]
    let wide_input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    let (wide_lo, wide_hi) = bounds_min_max(&wide_output);
    let wide_width = wide_hi - wide_lo;

    // Narrow input: [0.25, 0.75]
    let narrow_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CH, IMG_SIZE, IMG_SIZE]), 0.25f32),
        ArrayD::from_elem(IxDyn(&[IN_CH, IMG_SIZE, IMG_SIZE]), 0.75f32),
    )
    .expect("valid narrow bounds");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");
    let (narrow_lo, narrow_hi) = bounds_min_max(&narrow_output);
    let narrow_width = narrow_hi - narrow_lo;

    eprintln!(
        "Monotone tightening: wide=[{wide_lo}, {wide_hi}] width={wide_width:.6}, \
         narrow=[{narrow_lo}, {narrow_hi}] width={narrow_width:.6}"
    );

    // Narrower input should produce narrower (or equal) output bounds
    assert!(
        narrow_width <= wide_width + 1e-6,
        "monotone tightening: narrow width {narrow_width:.6} should be <= wide width {wide_width:.6}"
    );
}
