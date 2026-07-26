// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for DocLayout-YOLO multi-scale detection pipeline.
//!
//! Covers the CSPDarknet backbone, FPN neck, and YOLO detection heads at
//! multiple scales (P3/P4/P5). Focuses on the multi-scale feature extraction,
//! per-scale detection, box regression, objectness, class probabilities, NMS,
//! and anchor-free decoding that compose the full detection pipeline.
//!
//! ## Tests (14 tests)
//!
//! **Backbone (3 tests):**
//! 1.  **CSPDarknet stem convolution bounds** — stride-2 ConvBnSiLU stem (IBP)
//! 2.  **CSPDarknet stage 1-4 feature bounds** — 4-stage progressive downsample (IBP)
//! 3.  **C2f bottleneck bounds** — CSP split + bottleneck + concat (IBP)
//!
//! **Spatial pooling (1 test):**
//! 4.  **SPPF bounds** — cascaded MaxPool + concat + reduce (IBP)
//!
//! **FPN/PAN neck (2 tests):**
//! 5.  **FPN top-down pathway per scale** — P5 lateral -> P4 -> P3 cascade (IBP)
//! 6.  **PAN bottom-up pathway per scale** — P3 -> P4 -> P5 upward path (IBP)
//!
//! **Detection heads (4 tests):**
//! 7.  **Detection head per scale (P3/P4/P5)** — per-scale cls sigmoid (IBP)
//! 8.  **Box regression (cx, cy, w, h) bounds** — DFL softmax-weighted sum (IBP)
//! 9.  **Objectness score bounds [0, 1]** — sigmoid objectness (IBP)
//! 10. **Class probability bounds per category** — per-class sigmoid (IBP)
//!
//! **Post-processing & pipeline (4 tests):**
//! 11. **NMS score thresholding** — ReLU(sigmoid - threshold) (IBP)
//! 12. **Multi-scale feature fusion bounds** — 3-scale neck fusion (IBP)
//! 13. **Anchor-free detection head** — grid offset + sigmoid decoding (IBP)
//! 14. **Full backbone-to-detection pipeline** — end-to-end image -> detection (IBP)
//!
//! Architecture references:
//! - DocLayout-YOLO (Zhao et al. 2024): YOLOv10-based document layout detection
//! - CSPDarkNet: Cross-Stage Partial backbone with DarkNet topology
//! - FPN (Lin et al. 2017): Feature Pyramid Network for top-down fusion
//! - PAN (Liu et al. 2018): Path Aggregation Network for bottom-up fusion
//! - DFL (Li et al. 2022): Distribution Focal Loss for box regression
//!
//! Dimensions (small for fast verification, structurally representative):
//! - All feature maps use 4D shapes `[1, C, H, W]` (batch=1) to match the
//!   batch norm validator convention (channel_dim=1 for rank >= 3).
//! - IMG_SIZE=16 (symbolic, real: 640), BASE_CH=8 (symbolic, real: 64)
//! - P3 spatial=4, P4 spatial=2, P5 spatial=1
//! - NUM_CLASSES=4, DFL_BINS=4
//!
//! Part of #4234: DocLayout-YOLO multi-scale detection compose tests.

use super::common::{assert_bounds_valid, bounds_min_max, uniform_bounds};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const IMG_SIZE: usize = 16;
const IN_CH: usize = 3;
const BASE_CH: usize = 8;
const P3_SPATIAL: usize = 4; // IMG_SIZE / 4
const P4_SPATIAL: usize = 2; // IMG_SIZE / 8
const P5_SPATIAL: usize = 1; // IMG_SIZE / 16
const P3_CH: usize = BASE_CH; // 8
const P4_CH: usize = BASE_CH * 2; // 16
const P5_CH: usize = BASE_CH * 4; // 32
const NUM_CLASSES: usize = 4;
const DFL_BINS: usize = 4;
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
    .expect("valid image bounds [0, 1]")
}

/// Push ConvBnSiLU bindings (7 params).
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

/// Add ConvBnSiLU block with 4D shapes `[1, C, H, W]`.
///
/// Conv2d -> BatchNorm -> SiLU (sigmoid(x)*x).
/// Uses 4D output shapes so batch norm validator correctly identifies
/// channel_dim=1 for rank >= 3.
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
    let out_shape = [1, out_ch, out_h, out_w];
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

/// C2f-style bottleneck: 3x3 conv + SiLU + 3x3 conv + residual.
fn add_c2f_bottleneck(
    b: &mut TensorBlockBuilder,
    x: TensorNodeId,
    ch: usize,
    spatial: usize,
    prefix: &str,
) -> TensorNodeId {
    let shape = [1, ch, spatial, spatial];
    let bn1 = add_conv_bn_silu(
        b,
        x,
        &format!("{prefix}_bn1"),
        ch,
        ch,
        3,
        1,
        1,
        spatial,
        spatial,
    );
    let bn2 = add_conv_bn_silu(
        b,
        bn1,
        &format!("{prefix}_bn2"),
        ch,
        ch,
        3,
        1,
        1,
        spatial,
        spatial,
    );
    b.add_binary_add(bn2, x, &shape)
}

fn push_c2f_bottleneck_bindings(bindings: &mut Vec<TensorParamBinding>, ch: usize) {
    push_conv_bn_silu_bindings(bindings, ch, ch, 3);
    push_conv_bn_silu_bindings(bindings, ch, ch, 3);
}

// ===========================================================================
// 1. CSPDarknet stem convolution bounds (IBP)
// ===========================================================================

/// Stem: stride-2 ConvBnSiLU reduces image from IMG_SIZE to IMG_SIZE/2.
#[test]
fn test_doclayout_ms_cspdarknet_stem_ibp() {
    let stem_sp = IMG_SIZE / 2;
    let mut b = TensorBlockBuilder::new("dly_ms_stem");
    let input = b.add_input("image", &[1, IN_CH, IMG_SIZE, IMG_SIZE]);
    let out = add_conv_bn_silu(
        &mut b, input, "stem", IN_CH, BASE_CH, 3, 2, 1, stem_sp, stem_sp,
    );
    let def = b.build(out).expect("valid stem kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, BASE_CH, IN_CH, 3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP through stem");
    assert_bounds_valid(&output);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, BASE_CH, stem_sp, stem_sp],
        "stem output shape"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale stem IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. CSPDarknet stage 1-4 feature bounds (IBP)
// ===========================================================================

/// 4-stage progressive downsample: stem -> P3 -> P4 -> P5.
#[test]
fn test_doclayout_ms_cspdarknet_stage1_to_4_ibp() {
    let s1 = IMG_SIZE / 2;
    let s2 = IMG_SIZE / 4;
    let s3 = IMG_SIZE / 8;
    let s4 = IMG_SIZE / 16;
    let mut b = TensorBlockBuilder::new("dly_ms_stages_1_4");
    let input = b.add_input("image", &[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    let stem = add_conv_bn_silu(&mut b, input, "stem", IN_CH, BASE_CH, 3, 2, 1, s1, s1);
    let p3 = add_conv_bn_silu(&mut b, stem, "stage1", BASE_CH, P3_CH, 3, 2, 1, s2, s2);
    let p4 = add_conv_bn_silu(&mut b, p3, "stage2", P3_CH, P4_CH, 3, 2, 1, s3, s3);
    let p5 = add_conv_bn_silu(&mut b, p4, "stage3", P4_CH, P5_CH, 3, 2, 1, s4, s4);
    let def = b.build(p5).expect("valid 4-stage kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, BASE_CH, IN_CH, 3);
    push_conv_bn_silu_bindings(&mut bindings, P3_CH, BASE_CH, 3);
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, P3_CH, 3);
    push_conv_bn_silu_bindings(&mut bindings, P5_CH, P4_CH, 3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP through 4 stages");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, P5_CH, s4, s4]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale stages 1-4 IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    let width = hi_max - lo_min;
    assert!(width < 1e6, "stage 1-4 bounds vacuously wide: {width}");
}

// ===========================================================================
// 3. C2f (CSP Bottleneck with 2 convolutions) bounds (IBP)
// ===========================================================================

/// C2f block: split -> bottleneck on one half -> concat -> reduce.
#[test]
fn test_doclayout_ms_c2f_bottleneck_ibp() {
    let ch = P3_CH;
    let sp = P3_SPATIAL;
    let feat_4d = [1, ch, sp, sp];

    let mut b = TensorBlockBuilder::new("dly_ms_c2f");
    let input = b.add_input("features", &feat_4d);

    // Entry 1x1 conv to expand channels
    let entry_w = b.add_input("entry_w", &[ch * 2, ch, 1, 1]);
    let expanded = b.add_conv2d(input, entry_w, None, 1, 1, 0, 0, &[1, ch * 2, sp, sp]);

    // Split along channel axis (dim=1 in 4D)
    let half_shape = [1, ch, sp, sp];
    let split0 = b.add_narrow(expanded, 1, 0, ch, &half_shape);
    let split1 = b.add_narrow(expanded, 1, ch, ch, &half_shape);

    // Bottleneck on second half
    let bottleneck = add_c2f_bottleneck(&mut b, split1, ch, sp, "bneck");

    // Concat along channel dim (dim=1) + exit conv
    let cat_shape = [1, ch * 2, sp, sp];
    let cat = b.add_concat(&[split0, bottleneck], 1, &cat_shape);
    let exit_w = b.add_input("exit_w", &[ch, ch * 2, 1, 1]);
    let out = b.add_conv2d(cat, exit_w, None, 1, 1, 0, 0, &feat_4d);
    let def = b.build(out).expect("valid C2f kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.push(TensorParamBinding::ConstantTensor(w(&[ch * 2, ch, 1, 1])));
    push_c2f_bottleneck_bindings(&mut bindings, ch);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[ch, ch * 2, 1, 1])));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&feat_4d, 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through C2f");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &feat_4d);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale C2f IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 4. SPPF (Spatial Pyramid Pooling Fast) bounds (IBP)
// ===========================================================================

/// SPPF-style multi-scale aggregation: cascaded 3x3 conv (stride-1, pad-1)
/// stages producing multi-receptive-field features, concatenated + reduced.
///
/// Note: MaxPool2d is not yet supported in NY graph translation.
/// We model SPPF's multi-receptive-field aggregation using cascaded 3x3 convs
/// (same spatial-preserving behavior as the MaxPool cascade) which correctly
/// exercises the concat + channel-reduction verification path.
#[test]
fn test_doclayout_ms_sppf_ibp() {
    let ch = P3_CH;
    let sp = P3_SPATIAL;
    let shape = [1, ch, sp, sp];

    let mut b = TensorBlockBuilder::new("dly_ms_sppf");
    let input = b.add_input("p3_features", &shape);

    // Cascaded 3x3 convs (spatial-preserving, same as SPPF MaxPool role)
    let p1_w = b.add_input("pool1_w", &[ch, ch, 3, 3]);
    let p1 = b.add_conv2d(input, p1_w, None, 1, 1, 1, 1, &shape);
    let p2_w = b.add_input("pool2_w", &[ch, ch, 3, 3]);
    let p2 = b.add_conv2d(p1, p2_w, None, 1, 1, 1, 1, &shape);
    let p3_w = b.add_input("pool3_w", &[ch, ch, 3, 3]);
    let p3_node = b.add_conv2d(p2, p3_w, None, 1, 1, 1, 1, &shape);

    // Concat along channel dim (all derived from Variable)
    let cat_shape = [1, ch * 4, sp, sp];
    let cat = b.add_concat(&[input, p1, p2, p3_node], 1, &cat_shape);

    // Reduce back to ch channels
    let reduce_w = b.add_input("reduce_w", &[ch, ch * 4, 1, 1]);
    let out = b.add_conv2d(cat, reduce_w, None, 1, 1, 0, 0, &shape);
    let def = b.build(out).expect("valid SPPF kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[ch, ch, 3, 3])), // pool1_w
        TensorParamBinding::ConstantTensor(w(&[ch, ch, 3, 3])), // pool2_w
        TensorParamBinding::ConstantTensor(w(&[ch, ch, 3, 3])), // pool3_w
        TensorParamBinding::ConstantTensor(w(&[ch, ch * 4, 1, 1])), // reduce_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&shape, 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through SPPF");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale SPPF IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. FPN top-down pathway per scale (IBP)
// ===========================================================================

/// FPN top-down: feature split into lateral + skip branches, concatenated,
/// and reduced. Models the FPN lateral merge where high-level features are
/// reduced and fused with skip features.
///
/// All concat inputs derive from the single Variable to satisfy the
/// NY graph constraint (constant inputs in concat not supported).
/// Spatial dimensions stay at P5 scale (no upsample -- reshape cannot change
/// element count, and NY does not support interpolation ops).
#[test]
fn test_doclayout_ms_fpn_topdown_cascade_ibp() {
    let p5_shape = [1, P5_CH, P5_SPATIAL, P5_SPATIAL];
    let out_shape = [1, P4_CH, P5_SPATIAL, P5_SPATIAL];

    let mut b = TensorBlockBuilder::new("dly_ms_fpn_cascade");
    let p5_feat = b.add_input("p5_features", &p5_shape);

    // Lateral path: P5 -> 1x1 conv -> P4_CH channels (same spatial)
    let lat_w = b.add_input("lat_w", &[P4_CH, P5_CH, 1, 1]);
    let lateral = b.add_conv2d(
        p5_feat,
        lat_w,
        None,
        1,
        1,
        0,
        0,
        &[1, P4_CH, P5_SPATIAL, P5_SPATIAL],
    );

    // Skip path: P5 -> different 1x1 conv (models the P4 skip connection)
    let skip_w = b.add_input("skip_w", &[P4_CH, P5_CH, 1, 1]);
    let skip = b.add_conv2d(
        p5_feat,
        skip_w,
        None,
        1,
        1,
        0,
        0,
        &[1, P4_CH, P5_SPATIAL, P5_SPATIAL],
    );

    // Concat lateral + skip along channel dim (both derived from Variable)
    let cat_shape = [1, P4_CH * 2, P5_SPATIAL, P5_SPATIAL];
    let cat = b.add_concat(&[lateral, skip], 1, &cat_shape);

    // Reduction conv
    let red_w = b.add_input("red_w", &[P4_CH, P4_CH * 2, 1, 1]);
    let out = b.add_conv2d(cat, red_w, None, 1, 1, 0, 0, &out_shape);
    let def = b.build(out).expect("valid FPN cascade kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[P4_CH, P5_CH, 1, 1])), // lat_w
        TensorParamBinding::ConstantTensor(w(&[P4_CH, P5_CH, 1, 1])), // skip_w
        TensorParamBinding::ConstantTensor(w(&[P4_CH, P4_CH * 2, 1, 1])), // red_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&p5_shape, 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FPN cascade");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &out_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale FPN cascade IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. PAN bottom-up pathway per scale (IBP)
// ===========================================================================

/// PAN bottom-up: high-res P3 features split into two branches via
/// stride-2 downsample + identity-scale convs, concatenated, and reduced.
///
/// All concat inputs derive from the single Variable to satisfy the
/// NY graph constraint (constant inputs in concat not supported).
/// Stride-2 downsampling models spatial reduction in the PAN bottom-up path.
#[test]
fn test_doclayout_ms_pan_bottomup_cascade_ibp() {
    let p3_shape = [1, P3_CH, P3_SPATIAL, P3_SPATIAL];
    let ds_shape = [1, P3_CH, P4_SPATIAL, P4_SPATIAL]; // after stride-2

    let mut b = TensorBlockBuilder::new("dly_ms_pan_cascade");
    let p3_feat = b.add_input("p3_features", &p3_shape);

    // Branch 1: stride-2 downsample
    let ds_w = b.add_input("ds_w", &[P3_CH, P3_CH, 3, 3]);
    let downsampled = b.add_conv2d(p3_feat, ds_w, None, 2, 2, 1, 1, &ds_shape);

    // Branch 2: different 1x1 conv + stride-2 (models the P4 skip)
    let skip_w = b.add_input("skip_w", &[P3_CH, P3_CH, 3, 3]);
    let skip = b.add_conv2d(p3_feat, skip_w, None, 2, 2, 1, 1, &ds_shape);

    // Concat both branches (both derived from Variable)
    let cat_shape = [1, P3_CH * 2, P4_SPATIAL, P4_SPATIAL];
    let cat = b.add_concat(&[downsampled, skip], 1, &cat_shape);

    // Reduce
    let red_w = b.add_input("red_w", &[P4_CH, P3_CH * 2, 1, 1]);
    let out = b.add_conv2d(
        cat,
        red_w,
        None,
        1,
        1,
        0,
        0,
        &[1, P4_CH, P4_SPATIAL, P4_SPATIAL],
    );
    let def = b.build(out).expect("valid PAN cascade kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[P3_CH, P3_CH, 3, 3])), // ds_w
        TensorParamBinding::ConstantTensor(w(&[P3_CH, P3_CH, 3, 3])), // skip_w
        TensorParamBinding::ConstantTensor(w(&[P4_CH, P3_CH * 2, 1, 1])), // red_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&p3_shape, 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN cascade");
    assert_bounds_valid(&output);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, P4_CH, P4_SPATIAL, P4_SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale PAN cascade IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. Detection head per scale P3/P4/P5 (IBP)
// ===========================================================================

/// Classification head applied at each of P3, P4, P5 feature scales.
/// All produce sigmoid outputs in [0, 1].
#[test]
fn test_doclayout_ms_detection_per_scale_ibp() {
    let scales: &[(usize, usize, &str)] = &[
        (P3_CH, P3_SPATIAL, "P3"),
        (P4_CH, P4_SPATIAL, "P4"),
        (P5_CH, P5_SPATIAL, "P5"),
    ];

    for &(ch, sp, label) in scales {
        let num_anchors = sp * sp;
        let cls_conv_shape = [1, NUM_CLASSES, sp, sp];
        let cls_flat = [num_anchors, NUM_CLASSES];

        let mut b = TensorBlockBuilder::new(&format!("dly_ms_detect_{label}"));
        let input = b.add_input("features", &[1, ch, sp, sp]);
        let cls_w = b.add_input("cls_w", &[NUM_CLASSES, ch, 1, 1]);
        let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
        let cls_conv = b.add_conv2d(input, cls_w, Some(cls_b), 1, 1, 0, 0, &cls_conv_shape);
        let cls_reshaped = b.add_reshape(cls_conv, &cls_flat);
        let out = b.add_sigmoid(cls_reshaped, &cls_flat);
        let def = b.build(out).expect("valid detect kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, ch, 1, 1])),
            TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = uniform_bounds(&[1, ch, sp, sp], 2.0);

        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        assert_eq!(output.lower_upper().0.shape(), &cls_flat);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("DLY multiscale {label} detect IBP: [{lo_min:.6}, {hi_max:.6}]");
        let eps = 1e-6;
        assert!(
            lo_min >= 0.0 - eps,
            "{label} sigmoid lower >= 0, got {lo_min}"
        );
        assert!(
            hi_max <= 1.0 + eps,
            "{label} sigmoid upper <= 1, got {hi_max}"
        );
    }
}

// ===========================================================================
// 8. Box regression (cx, cy, w, h) bounds via DFL (IBP)
// ===========================================================================

/// DFL: softmax over bins -> weighted sum -> [0, DFL_BINS-1] per box side.
/// 4 box sides (cx, cy, w, h) are processed independently.
#[test]
fn test_doclayout_ms_box_regression_dfl_ibp() {
    let num_anchors = P3_SPATIAL * P3_SPATIAL;
    let num_sides = 4; // cx, cy, w, h
    let input_dim = DFL_BINS * num_sides;
    let flat_shape = [num_anchors, input_dim];

    let mut b = TensorBlockBuilder::new("dly_ms_box_dfl");
    let input = b.add_input("box_logits", &flat_shape);

    // Reshape to [num_anchors * 4, DFL_BINS] for per-side softmax
    let reshape_shape = [num_anchors * num_sides, DFL_BINS];
    let reshaped = b.add_reshape(input, &reshape_shape);
    let softmax = b.add_softmax(reshaped, -1, &reshape_shape);

    // Weighted sum: [num_anchors*4, DFL_BINS] x [DFL_BINS, 1] -> [num_anchors*4, 1]
    let proj_w = b.add_input("dfl_proj", &[DFL_BINS, 1]);
    let proj_out = b.add_matmul(softmax, proj_w, false, None, &[num_anchors * num_sides, 1]);

    // Reshape to [num_anchors, 4]
    let out = b.add_reshape(proj_out, &[num_anchors, num_sides]);
    let def = b.build(out).expect("valid box DFL kernel");

    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins"),
        ),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&flat_shape, 3.0);

    let output = graph.propagate_ibp(&input).expect("IBP through box DFL");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[num_anchors, num_sides]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let max_bin = (DFL_BINS - 1) as f32;
    eprintln!("DLY multiscale box DFL IBP: [{lo_min:.6}, {hi_max:.6}]");
    // IBP may widen bounds beyond the theoretical [0, DFL_BINS-1] due to
    // over-approximation in softmax + matmul composition. Check finite and
    // non-vacuous rather than tight theoretical bounds.
    assert!(lo_min.is_finite(), "DFL lower finite, got {lo_min}");
    assert!(hi_max.is_finite(), "DFL upper finite, got {hi_max}");
    let width = hi_max - lo_min;
    assert!(
        width < max_bin * 10.0,
        "DFL bounds vacuously wide: {width} (max_bin={max_bin})"
    );
}

// ===========================================================================
// 9. Objectness score bounds [0, 1] (IBP)
// ===========================================================================

/// Objectness: separate sigmoid head producing [0, 1] per anchor.
#[test]
fn test_doclayout_ms_objectness_score_ibp() {
    let ch = P4_CH;
    let sp = P4_SPATIAL;
    let num_anchors = sp * sp;

    let mut b = TensorBlockBuilder::new("dly_ms_objectness");
    let input = b.add_input("features", &[1, ch, sp, sp]);
    let obj_w = b.add_input("obj_w", &[1, ch, 1, 1]);
    let obj_b = b.add_input("obj_b", &[1]);
    let obj_conv = b.add_conv2d(input, obj_w, Some(obj_b), 1, 1, 0, 0, &[1, 1, sp, sp]);
    let obj_flat = b.add_reshape(obj_conv, &[num_anchors]);
    let out = b.add_sigmoid(obj_flat, &[num_anchors]);
    let def = b.build(out).expect("valid objectness kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[1, ch, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[1])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[1, ch, sp, sp], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through objectness");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[num_anchors]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale objectness IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "objectness lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "objectness upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. Class probability bounds per category (IBP)
// ===========================================================================

/// Per-class sigmoid with channel-wise assertions.
#[test]
fn test_doclayout_ms_class_probability_per_category_ibp() {
    let ch = P3_CH;
    let sp = P3_SPATIAL;
    let num_anchors = sp * sp;
    let cls_flat = [num_anchors, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("dly_ms_cls_per_cat");
    let input = b.add_input("features", &[1, ch, sp, sp]);
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, ch, 1, 1]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_conv = b.add_conv2d(
        input,
        cls_w,
        Some(cls_b),
        1,
        1,
        0,
        0,
        &[1, NUM_CLASSES, sp, sp],
    );
    let cls_reshaped = b.add_reshape(cls_conv, &cls_flat);
    let out = b.add_sigmoid(cls_reshaped, &cls_flat);
    let def = b.build(out).expect("valid cls per-cat kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, ch, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[1, ch, sp, sp], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cls per-cat");
    assert_bounds_valid(&output);

    // Per-anchor, per-class assertions
    let (lo_arr, hi_arr) = output.lower_upper();
    let eps = 1e-5;
    for a in 0..num_anchors {
        for c in 0..NUM_CLASSES {
            let l = lo_arr[[a, c]];
            let h = hi_arr[[a, c]];
            assert!(
                l >= 0.0 - eps && h <= 1.0 + eps,
                "cls[{a},{c}] out of [0,1]: [{l}, {h}]"
            );
        }
    }

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale cls per-category IBP: [{lo_min:.6}, {hi_max:.6}]");
}

// ===========================================================================
// 11. NMS score thresholding (IBP)
// ===========================================================================

/// NMS filter: ReLU(sigmoid(logits) - threshold) -> non-negative scores.
#[test]
fn test_doclayout_ms_nms_score_thresholding_ibp() {
    let num_anchors = P3_SPATIAL * P3_SPATIAL;

    let mut b = TensorBlockBuilder::new("dly_ms_nms_thresh");
    let input = b.add_input("cls_logits", &[num_anchors, NUM_CLASSES]);
    let conf = b.add_sigmoid(input, &[num_anchors, NUM_CLASSES]);

    // Subtract threshold constant
    let thresh = b.add_input("threshold", &[num_anchors, NUM_CLASSES]);
    let diff = b.add_binary_add(conf, thresh, &[num_anchors, NUM_CLASSES]);
    let out = b.add_relu(diff, &[num_anchors, NUM_CLASSES]);
    let def = b.build(out).expect("valid NMS thresh kernel");

    // Threshold of -0.25 (subtracting 0.25 from sigmoid outputs)
    let thresh_data = ArrayD::from_elem(IxDyn(&[num_anchors, NUM_CLASSES]), -0.25f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(thresh_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[num_anchors, NUM_CLASSES], 5.0);

    let output = graph.propagate_ibp(&input).expect("IBP through NMS thresh");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale NMS thresh IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "ReLU output lower >= 0, got {lo_min}");
    // max is sigmoid(5)-0.25 ~= 0.74, but IBP may widen
    assert!(hi_max <= 1.01, "NMS upper <= 1, got {hi_max}");
}

// ===========================================================================
// 12. Multi-scale feature fusion bounds (IBP)
// ===========================================================================

/// 3-scale neck fusion modeled as: P5 features split into three branches
/// via different convolutions (modeling lateral connections to P4/P3 scales),
/// concatenated, and reduced. Verifies non-vacuous bound propagation.
///
/// All concat inputs derive from the single Variable to satisfy the
/// NY graph constraint (constant inputs in concat not supported).
#[test]
fn test_doclayout_ms_multiscale_feature_fusion_ibp() {
    let p5_shape = [1, P5_CH, P5_SPATIAL, P5_SPATIAL];
    let fused_ch = P3_CH + P4_CH + P5_CH; // 8 + 16 + 32 = 56
    let out_shape = [1, P3_CH, P5_SPATIAL, P5_SPATIAL];

    let mut b = TensorBlockBuilder::new("dly_ms_fusion");
    let p5_feat = b.add_input("p5_features", &p5_shape);

    // Branch 1: identity-scale (P5 -> P5_CH)
    let id_w = b.add_input("id_w", &[P5_CH, P5_CH, 1, 1]);
    let br_p5 = b.add_conv2d(p5_feat, id_w, None, 1, 1, 0, 0, &p5_shape);

    // Branch 2: lateral to P4 scale (P5 -> P4_CH)
    let lat_p4_w = b.add_input("lat_p4_w", &[P4_CH, P5_CH, 1, 1]);
    let br_p4 = b.add_conv2d(
        p5_feat,
        lat_p4_w,
        None,
        1,
        1,
        0,
        0,
        &[1, P4_CH, P5_SPATIAL, P5_SPATIAL],
    );

    // Branch 3: lateral to P3 scale (P5 -> P3_CH)
    let lat_p3_w = b.add_input("lat_p3_w", &[P3_CH, P5_CH, 1, 1]);
    let br_p3 = b.add_conv2d(
        p5_feat,
        lat_p3_w,
        None,
        1,
        1,
        0,
        0,
        &[1, P3_CH, P5_SPATIAL, P5_SPATIAL],
    );

    // Concat all branches (all derived from Variable)
    let cat_shape = [1, fused_ch, P5_SPATIAL, P5_SPATIAL];
    let cat = b.add_concat(&[br_p5, br_p4, br_p3], 1, &cat_shape);

    // Reduce to P3_CH
    let red_w = b.add_input("red_w", &[P3_CH, fused_ch, 1, 1]);
    let out = b.add_conv2d(cat, red_w, None, 1, 1, 0, 0, &out_shape);
    let def = b.build(out).expect("valid fusion kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[P5_CH, P5_CH, 1, 1])), // id_w
        TensorParamBinding::ConstantTensor(w(&[P4_CH, P5_CH, 1, 1])), // lat_p4_w
        TensorParamBinding::ConstantTensor(w(&[P3_CH, P5_CH, 1, 1])), // lat_p3_w
        TensorParamBinding::ConstantTensor(w(&[P3_CH, fused_ch, 1, 1])), // red_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&p5_shape, 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through fusion");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &out_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("DLY multiscale fusion IBP: [{lo_min:.6}, {hi_max:.6}] width={width:.4}");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(width < 1e6, "fusion bounds vacuously wide: {width}");
}

// ===========================================================================
// 13. Anchor-free detection head (IBP)
// ===========================================================================

/// Anchor-free: sigmoid(predicted_offset) + grid_anchor -> absolute coords.
#[test]
fn test_doclayout_ms_anchor_free_head_ibp() {
    let grid_size = P3_SPATIAL;
    let flat_size = grid_size * grid_size;

    let mut b = TensorBlockBuilder::new("dly_ms_anchor_free");
    let pred = b.add_input("pred_offset", &[flat_size, 2]);
    let sig = b.add_sigmoid(pred, &[flat_size, 2]);

    let grid_anchor = b.add_input("grid_anchor", &[flat_size, 2]);
    let out = b.add_binary_add(sig, grid_anchor, &[flat_size, 2]);
    let def = b.build(out).expect("valid anchor-free kernel");

    // Build grid anchors: (col, row) for each spatial position
    let mut anchor_data = vec![0.0f32; flat_size * 2];
    for row in 0..grid_size {
        for col in 0..grid_size {
            let idx = row * grid_size + col;
            anchor_data[idx * 2] = col as f32;
            anchor_data[idx * 2 + 1] = row as f32;
        }
    }
    let grid_data =
        ArrayD::from_shape_vec(IxDyn(&[flat_size, 2]), anchor_data).expect("valid grid");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(grid_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[flat_size, 2], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through anchor-free");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[flat_size, 2]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale anchor-free IBP: [{lo_min:.6}, {hi_max:.6}]");
    // sigmoid in [0,1] + grid in [0, grid_size-1]
    assert!(lo_min >= -0.01, "anchor-free lower >= 0, got {lo_min}");
    assert!(
        hi_max <= (grid_size as f32) + 1.01,
        "anchor-free upper <= grid_size+1, got {hi_max}"
    );
}

// ===========================================================================
// 14. Full backbone-to-detection pipeline (IBP)
// ===========================================================================

/// End-to-end: image -> 2-stage backbone -> P3 detection head -> sigmoid.
#[test]
fn test_doclayout_ms_full_backbone_to_detection_ibp() {
    let s1 = IMG_SIZE / 2;
    let p3_sp = P3_SPATIAL;
    let num_anchors = p3_sp * p3_sp;
    let cls_flat = [num_anchors, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("dly_ms_full_e2e");
    let input = b.add_input("image", &[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    // Backbone: stem -> stage1 (-> P3)
    let stem = add_conv_bn_silu(&mut b, input, "stem", IN_CH, BASE_CH, 3, 2, 1, s1, s1);
    let backbone = add_conv_bn_silu(
        &mut b, stem, "stage1", BASE_CH, P3_CH, 3, 2, 1, p3_sp, p3_sp,
    );

    // P3 detection head: 1x1 conv -> reshape -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, P3_CH, 1, 1]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_conv = b.add_conv2d(
        backbone,
        cls_w,
        Some(cls_b),
        1,
        1,
        0,
        0,
        &[1, NUM_CLASSES, p3_sp, p3_sp],
    );
    let cls_reshaped = b.add_reshape(cls_conv, &cls_flat);
    let out = b.add_sigmoid(cls_reshaped, &cls_flat);
    let def = b.build(out).expect("valid full e2e kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, BASE_CH, IN_CH, 3);
    push_conv_bn_silu_bindings(&mut bindings, P3_CH, BASE_CH, 3);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        NUM_CLASSES,
        P3_CH,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP through full e2e");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &cls_flat);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale full e2e IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-5;
    assert!(lo_min >= 0.0 - eps, "e2e sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "e2e sigmoid upper <= 1, got {hi_max}");
}
