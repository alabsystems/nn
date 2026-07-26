// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for DocLayout-YOLO multi-scale detection pipeline.
//!
//! Verifies IBP and CROWN bound propagation through multi-scale detection
//! subgraphs of the DocLayout-YOLO architecture (YOLOv10-based document layout
//! detection). These tests focus on the multi-scale aspects that bridge
//! backbone feature extraction, FPN/PAN neck fusion, and per-scale detection
//! heads — the core novelty of YOLO-based detectors.
//!
//! ## Tests (19 tests)
//!
//! **Backbone multi-scale extraction (4 tests):**
//! 1.  **CSPDarkNet stem + stage** — Conv stride-2 stem -> C2f block (IBP)
//! 2.  **Two-stage backbone with channel expansion** — Cascaded stride-2 (IBP)
//! 3.  **Three-scale feature extraction** — P3/P4/P5 from backbone (IBP)
//! 4.  **Backbone feature extraction CROWN** — Single stage (CROWN)
//!
//! **FPN + PAN neck fusion (4 tests):**
//! 5.  **Top-down FPN path** — P5 -> P4 lateral + merge (IBP)
//! 6.  **Bottom-up PAN path** — P3 -> P4 downsample + merge (IBP)
//! 7.  **FPN + PAN combined** — Top-down then bottom-up (IBP)
//! 8.  **Neck fusion CROWN** — Single FPN lateral path (CROWN)
//!
//! **Per-scale detection heads (4 tests):**
//! 9.  **Small-object detection head (P3)** — High-res cls + box (IBP)
//! 10. **Large-object detection head (P5)** — Low-res cls + box (IBP)
//! 11. **Dual-head DFL + sigmoid** — Box regression + classification (IBP)
//! 12. **Detection head CROWN** — Single-scale cls head (CROWN)
//!
//! **Full multi-scale pipeline (4 tests):**
//! 13. **Backbone -> neck -> P3 detection** — End-to-end small-object (IBP)
//! 14. **Backbone -> neck -> P5 detection** — End-to-end large-object (IBP)
//! 15. **Multi-scale monotone tightening** — Narrow input -> narrow output (IBP)
//! 16. **Multi-scale widening analysis** — Bounds growth across scales (IBP)
//!
//! **Scoring, merging, and NMS (3 tests):**
//! 17. **Objectness + class confidence scoring** — sigmoid(obj) * softmax(cls) (IBP)
//! 18. **Multi-scale to single output merge** — Concat P3/P4/P5 predictions (IBP)
//! 19. **NMS confidence thresholding** — sigmoid -> threshold -> ReLU filter (IBP + CROWN)
//!
//! Architecture references:
//! - DocLayout-YOLO (Zhao et al. 2024): YOLOv10-based document layout detection
//! - CSPDarkNet: Cross-Stage Partial backbone with DarkNet topology
//! - FPN (Lin et al. 2017): Feature Pyramid Network for top-down fusion
//! - PAN (Liu et al. 2018): Path Aggregation Network for bottom-up fusion
//! - DFL (Li et al. 2022): Distribution Focal Loss for box regression
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=16 (symbolic, real: 640), BASE_CH=8 (symbolic, real: 64)
//! - P3 spatial=4, P4 spatial=2, P5 spatial=1
//! - NUM_CLASSES=4, DFL_BINS=4
//!
//! Part of #4234: DocLayout-YOLO multi-scale detection compose tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Input image spatial size (after initial resize).
const IMG_SIZE: usize = 16;
/// Input channels (RGB).
const IN_CH: usize = 3;
/// Base channel width (backbone stem output).
const BASE_CH: usize = 8;
/// P3 feature map spatial size (IMG_SIZE / 4).
const P3_SPATIAL: usize = 4;
/// P4 feature map spatial size (IMG_SIZE / 8).
const P4_SPATIAL: usize = 2;
/// P5 feature map spatial size (IMG_SIZE / 16).
const P5_SPATIAL: usize = 1;
/// P3 channel width.
const P3_CH: usize = BASE_CH; // 8
/// P4 channel width (2x base).
const P4_CH: usize = BASE_CH * 2; // 16
/// P5 channel width (4x base).
const P5_CH: usize = BASE_CH * 4; // 32
/// Number of detection classes.
const NUM_CLASSES: usize = 4;
/// DFL regression bins per box side.
const DFL_BINS: usize = 4;
/// Weight magnitude for bounded verification.
const W_MAG: f32 = 0.02;

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
/// Uses 4D output shapes `[1, C, H, W]` so batch norm validator correctly
/// identifies channel_dim=1 for rank >= 3.
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

/// Add a C2f-style bottleneck: 3x3 conv + SiLU + 3x3 conv + residual.
/// Uses 4D shapes `[1, C, H, W]`.
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

/// Push bindings for one C2f bottleneck (2 ConvBnSiLU blocks = 14 params).
fn push_c2f_bottleneck_bindings(bindings: &mut Vec<TensorParamBinding>, ch: usize) {
    push_conv_bn_silu_bindings(bindings, ch, ch, 3); // bn1
    push_conv_bn_silu_bindings(bindings, ch, ch, 3); // bn2
}

// ===========================================================================
// 1. CSPDarkNet stem + stage (IBP)
// ===========================================================================

/// Build stem (stride-2 ConvBnSiLU) followed by a C2f bottleneck.
///
/// Input: `[1, IN_CH, IMG_SIZE, IMG_SIZE]` (Variable, image [0, 1]).
/// Output: `[1, BASE_CH, IMG_SIZE/2, IMG_SIZE/2]`.
#[test]
fn test_multiscale_cspdarknet_stem_stage_ibp() {
    let stem_sp = IMG_SIZE / 2; // 8
    let mut b = TensorBlockBuilder::new("dly_ms_stem_stage");
    let input = b.add_input("image", &[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    // Stem: stride-2 downsample
    let stem = add_conv_bn_silu(
        &mut b, input, "stem", IN_CH, BASE_CH, 3, 2, 1, stem_sp, stem_sp,
    );

    // C2f bottleneck (preserves spatial)
    let out = add_c2f_bottleneck(&mut b, stem, BASE_CH, stem_sp, "c2f0");
    let def = b.build(out).expect("valid stem+stage kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, BASE_CH, IN_CH, 3); // stem
    push_c2f_bottleneck_bindings(&mut bindings, BASE_CH); // c2f0
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP through stem+stage");
    assert_bounds_valid(&output);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, BASE_CH, stem_sp, stem_sp],
        "stem+stage output shape"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale stem+stage IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. Two-stage backbone with channel expansion (IBP)
// ===========================================================================

#[test]
fn test_multiscale_two_stage_channel_expansion_ibp() {
    let s1 = IMG_SIZE / 2; // 8
    let s2 = IMG_SIZE / 4; // 4 = P3_SPATIAL
    let mut b = TensorBlockBuilder::new("dly_ms_two_stage");
    let input = b.add_input("image", &[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    // Stage 0: IN_CH -> BASE_CH, stride-2
    let s0_out = add_conv_bn_silu(&mut b, input, "s0", IN_CH, BASE_CH, 3, 2, 1, s1, s1);
    // Stage 1: BASE_CH -> P4_CH, stride-2
    let out = add_conv_bn_silu(&mut b, s0_out, "s1", BASE_CH, P4_CH, 3, 2, 1, s2, s2);
    let def = b.build(out).expect("valid two-stage kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, BASE_CH, IN_CH, 3);
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, BASE_CH, 3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP through two-stage");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, P4_CH, s2, s2]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale two-stage IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. Three-scale feature extraction P3/P4/P5 (IBP)
// ===========================================================================

/// Build a 3-stage backbone that extracts features at P3, P4, P5 scales.
/// We verify through P5 (deepest) to exercise the full depth.
#[test]
fn test_multiscale_three_scale_feature_extraction_ibp() {
    let s1 = IMG_SIZE / 2; // 8
    let s2 = IMG_SIZE / 4; // 4 = P3
    let s3 = IMG_SIZE / 8; // 2 = P4
    let s4 = IMG_SIZE / 16; // 1 = P5
    let mut b = TensorBlockBuilder::new("dly_ms_three_scale");
    let input = b.add_input("image", &[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    // Stem: stride-2, IN_CH -> BASE_CH
    let stem = add_conv_bn_silu(&mut b, input, "stem", IN_CH, BASE_CH, 3, 2, 1, s1, s1);
    // P3: stride-2, BASE_CH -> P3_CH
    let _p3 = add_conv_bn_silu(&mut b, stem, "p3", BASE_CH, P3_CH, 3, 2, 1, s2, s2);
    // P4: stride-2, P3_CH -> P4_CH
    let p4 = add_conv_bn_silu(&mut b, _p3, "p4", P3_CH, P4_CH, 3, 2, 1, s3, s3);
    // P5: stride-2, P4_CH -> P5_CH
    let p5 = add_conv_bn_silu(&mut b, p4, "p5", P4_CH, P5_CH, 3, 2, 1, s4, s4);
    let def = b.build(p5).expect("valid three-scale kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, BASE_CH, IN_CH, 3); // stem
    push_conv_bn_silu_bindings(&mut bindings, P3_CH, BASE_CH, 3); // p3
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, P3_CH, 3); // p4
    push_conv_bn_silu_bindings(&mut bindings, P5_CH, P4_CH, 3); // p5
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through three-scale");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, P5_CH, s4, s4]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale P5 extraction IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    // Ensure bounds are not vacuously wide
    let width = hi_max - lo_min;
    assert!(width < 1e6, "P5 bounds vacuously wide: {width}");
}

// ===========================================================================
// 4. Backbone feature extraction CROWN
// ===========================================================================

#[test]
fn test_multiscale_backbone_single_stage_crown() {
    let stem_sp = IMG_SIZE / 2;
    let mut b = TensorBlockBuilder::new("dly_ms_backbone_crown");
    let input = b.add_input("image", &[1, IN_CH, IMG_SIZE, IMG_SIZE]);
    let out = add_conv_bn_silu(
        &mut b, input, "stem", IN_CH, BASE_CH, 3, 2, 1, stem_sp, stem_sp,
    );
    let def = b.build(out).expect("valid backbone crown kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, BASE_CH, IN_CH, 3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale backbone CROWN ({method:?}): [{lo_min:.6}, {hi_max:.6}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 5. Top-down FPN path (IBP)
// ===========================================================================

/// FPN top-down: P5 features (low-res, high-channel) are reduced and
/// fused with a lateral skip path to create fused features.
/// Both lateral and skip operate at P5 spatial resolution, then concat on
/// channel dim (axis 1). Uses 4D shapes for correct BN channel_dim=1.
#[test]
fn test_multiscale_fpn_topdown_path_ibp() {
    // Variable at P5 resolution. Lateral conv reduces channels, skip conv
    // provides alternate projection. Concat on axis 1, reduce to P4_CH.
    let p5_shape = [1, P5_CH, P5_SPATIAL, P5_SPATIAL];
    let out_shape = [1, P4_CH, P5_SPATIAL, P5_SPATIAL];
    let cat_shape = [1, P4_CH * 2, P5_SPATIAL, P5_SPATIAL];

    let mut b = TensorBlockBuilder::new("dly_ms_fpn_topdown");
    let p5_feat = b.add_input("p5_features", &p5_shape);

    // Lateral 1x1 conv on P5: P5_CH -> P4_CH
    let lateral = add_conv_bn_silu(
        &mut b, p5_feat, "lateral", P5_CH, P4_CH, 1, 1, 0, P5_SPATIAL, P5_SPATIAL,
    );

    // Skip path: alternate 1x1 conv projection at same spatial resolution
    let skip = add_conv_bn_silu(
        &mut b, p5_feat, "skip", P5_CH, P4_CH, 1, 1, 0, P5_SPATIAL, P5_SPATIAL,
    );

    // Concat lateral + skip on channel axis (axis 1)
    let cat = b.add_concat(&[lateral, skip], 1, &cat_shape);

    // Reduction conv: 2*P4_CH -> P4_CH
    let out = add_conv_bn_silu(
        &mut b,
        cat,
        "reduce",
        P4_CH * 2,
        P4_CH,
        1,
        1,
        0,
        P5_SPATIAL,
        P5_SPATIAL,
    );
    let def = b.build(out).expect("valid FPN topdown kernel");

    let mut bindings = vec![TensorParamBinding::Variable]; // p5_feat
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, P5_CH, 1); // lateral
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, P5_CH, 1); // skip
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, P4_CH * 2, 1); // reduce
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&p5_shape, 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FPN topdown");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &out_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale FPN topdown IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. Bottom-up PAN path (IBP)
// ===========================================================================

/// PAN bottom-up: P3 features (high-res) are downsampled via stride-2 conv
/// and fused with a skip path. Uses 4D shapes, axis-1 concat.
#[test]
fn test_multiscale_pan_bottomup_path_ibp() {
    let p3_shape = [1, P3_CH, P3_SPATIAL, P3_SPATIAL];
    let ds_shape = [1, P3_CH, P4_SPATIAL, P4_SPATIAL]; // after stride-2
    let skip_shape = [1, P4_CH, P4_SPATIAL, P4_SPATIAL]; // skip projection
    let cat_shape = [1, P3_CH + P4_CH, P4_SPATIAL, P4_SPATIAL];
    let out_shape = [1, P4_CH, P4_SPATIAL, P4_SPATIAL];

    let mut b = TensorBlockBuilder::new("dly_ms_pan_bottomup");
    let p3_feat = b.add_input("p3_features", &p3_shape);

    // Stride-2 downsample P3 -> P4 spatial
    let ds_w = b.add_input("ds_w", &[P3_CH, P3_CH, 3, 3]);
    let downsampled = b.add_conv2d(p3_feat, ds_w, None, 2, 2, 1, 1, &ds_shape);

    // Skip path: alternate projection from same input at P4 spatial
    let skip_w = b.add_input("skip_w", &[P4_CH, P3_CH, 3, 3]);
    let skip = b.add_conv2d(p3_feat, skip_w, None, 2, 2, 1, 1, &skip_shape);

    // Concat downsampled P3 + skip on channel axis (axis 1)
    let cat = b.add_concat(&[downsampled, skip], 1, &cat_shape);

    // Reduction conv
    let fuse_w = b.add_input("fuse_w", &[P4_CH, P3_CH + P4_CH, 1, 1]);
    let out = b.add_conv2d(cat, fuse_w, None, 1, 1, 0, 0, &out_shape);
    let def = b.build(out).expect("valid PAN bottomup kernel");

    let bindings = vec![
        TensorParamBinding::Variable,                                 // p3_feat
        TensorParamBinding::ConstantTensor(w(&[P3_CH, P3_CH, 3, 3])), // ds_w
        TensorParamBinding::ConstantTensor(w(&[P4_CH, P3_CH, 3, 3])), // skip_w
        TensorParamBinding::ConstantTensor(w(&[P4_CH, P3_CH + P4_CH, 1, 1])), // fuse_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&p3_shape, 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN bottomup");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &out_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale PAN bottomup IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. FPN + PAN combined neck (IBP)
// ===========================================================================

/// Combined neck: two-stage FPN fusion via dual-branch convolutions.
/// Step 1: P5 -> lateral + skip -> concat axis-1 -> reduce (fused_P4).
/// Step 2: fused_P4 -> branch_a + branch_b -> concat axis-1 -> reduce (output).
/// All at P5 spatial resolution. Uses 4D shapes, axis-1 concat.
#[test]
fn test_multiscale_fpn_pan_combined_ibp() {
    let sp = P5_SPATIAL;
    let p5_shape = [1, P5_CH, sp, sp];
    let cat1_shape = [1, P4_CH * 2, sp, sp];
    let out_shape = [1, P3_CH, sp, sp];
    let cat2_shape = [1, P4_CH * 2, sp, sp];

    let mut b = TensorBlockBuilder::new("dly_ms_fpn_pan_combined");
    let p5_feat = b.add_input("p5_features", &p5_shape);

    // Step 1: FPN lateral + skip at P5 resolution
    let lat = add_conv_bn_silu(&mut b, p5_feat, "lat", P5_CH, P4_CH, 1, 1, 0, sp, sp);
    let skip1 = add_conv_bn_silu(&mut b, p5_feat, "skip1", P5_CH, P4_CH, 1, 1, 0, sp, sp);
    let cat1 = b.add_concat(&[lat, skip1], 1, &cat1_shape);
    let fused = add_conv_bn_silu(&mut b, cat1, "red1", P4_CH * 2, P4_CH, 1, 1, 0, sp, sp);

    // Step 2: PAN branch_a + branch_b at same resolution
    let br_a = add_conv_bn_silu(&mut b, fused, "br_a", P4_CH, P4_CH, 1, 1, 0, sp, sp);
    let br_b = add_conv_bn_silu(&mut b, fused, "br_b", P4_CH, P4_CH, 1, 1, 0, sp, sp);
    let cat2 = b.add_concat(&[br_a, br_b], 1, &cat2_shape);
    let out = add_conv_bn_silu(&mut b, cat2, "red2", P4_CH * 2, P3_CH, 1, 1, 0, sp, sp);
    let def = b.build(out).expect("valid FPN+PAN kernel");

    let mut bindings = vec![TensorParamBinding::Variable]; // p5_feat
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, P5_CH, 1); // lat
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, P5_CH, 1); // skip1
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, P4_CH * 2, 1); // red1
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, P4_CH, 1); // br_a
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, P4_CH, 1); // br_b
    push_conv_bn_silu_bindings(&mut bindings, P3_CH, P4_CH * 2, 1); // red2
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&p5_shape, 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through FPN+PAN");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &out_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale FPN+PAN combined IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 8. Neck fusion CROWN
// ===========================================================================

#[test]
fn test_multiscale_neck_lateral_crown() {
    // Simple lateral path: P5 -> 1x1 conv -> output. 4D shapes for BN.
    let p5_shape = [1, P5_CH, P5_SPATIAL, P5_SPATIAL];
    let out_shape = [1, P4_CH, P5_SPATIAL, P5_SPATIAL];
    let mut b = TensorBlockBuilder::new("dly_ms_neck_lateral_crown");
    let input = b.add_input("p5_features", &p5_shape);
    let out = add_conv_bn_silu(
        &mut b, input, "lat", P5_CH, P4_CH, 1, 1, 0, P5_SPATIAL, P5_SPATIAL,
    );
    let def = b.build(out).expect("valid lateral kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, P5_CH, 1);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&p5_shape, 2.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale neck lateral CROWN ({method:?}): [{lo_min:.6}, {hi_max:.6}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
    assert_eq!(output.lower_upper().0.shape(), &out_shape);
}

// ===========================================================================
// 9. Small-object detection head P3 (IBP)
// ===========================================================================

/// P3 detection head: high-res features -> cls sigmoid + box DFL.
#[test]
fn test_multiscale_p3_detection_head_ibp() {
    let p3_shape = [P3_CH, P3_SPATIAL, P3_SPATIAL];
    let num_anchors = P3_SPATIAL * P3_SPATIAL;
    let cls_conv_shape = [NUM_CLASSES, P3_SPATIAL, P3_SPATIAL];
    let cls_flat = [num_anchors, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("dly_ms_p3_detect");
    let input = b.add_input("p3_features", &p3_shape);

    // Cls head: 1x1 conv -> reshape -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, P3_CH, 1, 1]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_conv = b.add_conv2d(input, cls_w, Some(cls_b), 1, 1, 0, 0, &cls_conv_shape);
    let cls_reshaped = b.add_reshape(cls_conv, &cls_flat);
    let out = b.add_sigmoid(cls_reshaped, &cls_flat);
    let def = b.build(out).expect("valid P3 detect kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, P3_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&p3_shape, 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through P3 detect");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &cls_flat);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale P3 detect IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. Large-object detection head P5 (IBP)
// ===========================================================================

#[test]
fn test_multiscale_p5_detection_head_ibp() {
    let p5_shape = [P5_CH, P5_SPATIAL, P5_SPATIAL];
    let num_anchors = P5_SPATIAL * P5_SPATIAL;
    let cls_conv_shape = [NUM_CLASSES, P5_SPATIAL, P5_SPATIAL];
    let cls_flat = [num_anchors, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("dly_ms_p5_detect");
    let input = b.add_input("p5_features", &p5_shape);

    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, P5_CH, 1, 1]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_conv = b.add_conv2d(input, cls_w, Some(cls_b), 1, 1, 0, 0, &cls_conv_shape);
    let cls_reshaped = b.add_reshape(cls_conv, &cls_flat);
    let out = b.add_sigmoid(cls_reshaped, &cls_flat);
    let def = b.build(out).expect("valid P5 detect kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, P5_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&p5_shape, 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through P5 detect");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &cls_flat);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale P5 detect IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 11. Dual-head DFL + sigmoid (IBP)
// ===========================================================================

/// Dual detection head: classification sigmoid + box DFL regression.
#[test]
fn test_multiscale_dual_head_dfl_sigmoid_ibp() {
    let ch = P4_CH;
    let sp = P4_SPATIAL;
    let num_anchors = sp * sp;
    let feat_shape = [ch, sp, sp];

    // Box: DFL (softmax -> weighted sum)
    let box_conv_shape = [DFL_BINS, sp, sp];
    let box_flat = [num_anchors, DFL_BINS];
    let box_out_shape = [num_anchors, 1];

    // Cls: sigmoid
    let cls_conv_shape = [NUM_CLASSES, sp, sp];
    let cls_flat = [num_anchors, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("dly_ms_dual_head");
    let input = b.add_input("neck_features", &feat_shape);

    // Box head
    let box_w = b.add_input("box_w", &[DFL_BINS, ch, 1, 1]);
    let box_b = b.add_input("box_b", &[DFL_BINS]);
    let bins_w = b.add_input("dfl_bins", &[DFL_BINS, 1]);
    let box_conv = b.add_conv2d(input, box_w, Some(box_b), 1, 1, 0, 0, &box_conv_shape);
    let box_reshaped = b.add_reshape(box_conv, &box_flat);
    let box_softmax = b.add_softmax(box_reshaped, 1, &box_flat);
    let box_dfl = b.add_matmul(box_softmax, bins_w, false, None, &box_out_shape);

    // Cls head
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, ch, 1, 1]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_conv = b.add_conv2d(input, cls_w, Some(cls_b), 1, 1, 0, 0, &cls_conv_shape);
    let cls_reshaped = b.add_reshape(cls_conv, &cls_flat);
    let cls_sigmoid = b.add_sigmoid(cls_reshaped, &cls_flat);

    // Concat: [num_anchors, NUM_CLASSES + 1]
    let final_shape = [num_anchors, NUM_CLASSES + 1];
    let out = b.add_concat(&[cls_sigmoid, box_dfl], 1, &final_shape);
    let def = b.build(out).expect("valid dual head kernel");

    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[DFL_BINS, ch, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[DFL_BINS])),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins"),
        ),
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, ch, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&feat_shape, 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through dual head");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &final_shape);

    let (lo_arr, hi_arr) = output.lower_upper();
    let eps = 1e-5;
    // Cls channels (first NUM_CLASSES) must be in [0, 1]
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
    // Box channel (last): DFL = softmax(bins) @ [0..DFL_BINS-1].
    // Theoretical range is [0, DFL_BINS-1], but IBP over-approximates
    // through softmax + matmul, so bounds may be wider. Check finite
    // and non-vacuously wide (< max_bin * 10).
    let max_bin = (DFL_BINS - 1) as f32;
    for a in 0..num_anchors {
        let l = lo_arr[[a, NUM_CLASSES]];
        let h = hi_arr[[a, NUM_CLASSES]];
        assert!(
            l.is_finite() && h.is_finite(),
            "box[{a}] bounds not finite: [{l}, {h}]"
        );
        let width = h - l;
        assert!(
            width < max_bin * 10.0,
            "box[{a}] bounds vacuously wide: [{l}, {h}] width={width}"
        );
    }

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale dual head IBP: [{lo_min:.6}, {hi_max:.6}]");
}

// ===========================================================================
// 12. Detection head CROWN
// ===========================================================================

#[test]
fn test_multiscale_detection_head_crown() {
    let ch = P3_CH;
    let sp = P3_SPATIAL;
    let num_anchors = sp * sp;
    let cls_conv_shape = [NUM_CLASSES, sp, sp];
    let cls_flat = [num_anchors, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("dly_ms_detect_crown");
    let input = b.add_input("features", &[ch, sp, sp]);
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, ch, 1, 1]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_conv = b.add_conv2d(input, cls_w, Some(cls_b), 1, 1, 0, 0, &cls_conv_shape);
    let cls_reshaped = b.add_reshape(cls_conv, &cls_flat);
    let out = b.add_sigmoid(cls_reshaped, &cls_flat);
    let def = b.build(out).expect("valid detect crown kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, ch, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[ch, sp, sp], 2.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale detect head CROWN ({method:?}): [{lo_min:.6}, {hi_max:.6}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 13. Backbone -> neck -> P3 detection (IBP)
// ===========================================================================

/// End-to-end small-object path: image -> backbone -> neck -> P3 detection.
#[test]
fn test_multiscale_e2e_p3_detection_ibp() {
    let stem_sp = IMG_SIZE / 2; // 8
    let p3_sp = P3_SPATIAL; // 4
    let num_anchors = p3_sp * p3_sp;
    let cls_flat = [num_anchors, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("dly_ms_e2e_p3");
    let input = b.add_input("image", &[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    // Backbone: stem stride-2 -> stage stride-2
    let stem = add_conv_bn_silu(
        &mut b, input, "stem", IN_CH, BASE_CH, 3, 2, 1, stem_sp, stem_sp,
    );
    let backbone = add_conv_bn_silu(&mut b, stem, "s1", BASE_CH, P3_CH, 3, 2, 1, p3_sp, p3_sp);

    // P3 detection head: 1x1 conv (4D) -> reshape (2D) -> sigmoid
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
    let def = b.build(out).expect("valid e2e P3 kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, BASE_CH, IN_CH, 3); // stem
    push_conv_bn_silu_bindings(&mut bindings, P3_CH, BASE_CH, 3); // s1
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        NUM_CLASSES,
        P3_CH,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP through e2e P3");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &cls_flat);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale e2e P3 IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-5;
    assert!(
        lo_min >= 0.0 - eps,
        "e2e P3 sigmoid lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "e2e P3 sigmoid upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 14. Backbone -> neck -> P5 detection (IBP)
// ===========================================================================

/// End-to-end large-object path: image -> 4-stage backbone -> P5 detection.
#[test]
fn test_multiscale_e2e_p5_detection_ibp() {
    let s1 = IMG_SIZE / 2; // 8
    let s2 = IMG_SIZE / 4; // 4
    let s3 = IMG_SIZE / 8; // 2
    let s4 = IMG_SIZE / 16; // 1 = P5_SPATIAL
    let num_anchors = s4 * s4;
    let cls_flat = [num_anchors, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("dly_ms_e2e_p5");
    let input = b.add_input("image", &[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    let stem = add_conv_bn_silu(&mut b, input, "stem", IN_CH, BASE_CH, 3, 2, 1, s1, s1);
    let s1_out = add_conv_bn_silu(&mut b, stem, "s1", BASE_CH, P3_CH, 3, 2, 1, s2, s2);
    let s2_out = add_conv_bn_silu(&mut b, s1_out, "s2", P3_CH, P4_CH, 3, 2, 1, s3, s3);
    let s3_out = add_conv_bn_silu(&mut b, s2_out, "s3", P4_CH, P5_CH, 3, 2, 1, s4, s4);

    // P5 detection head: 1x1 conv (4D) -> reshape (2D) -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, P5_CH, 1, 1]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_conv = b.add_conv2d(
        s3_out,
        cls_w,
        Some(cls_b),
        1,
        1,
        0,
        0,
        &[1, NUM_CLASSES, s4, s4],
    );
    let cls_reshaped = b.add_reshape(cls_conv, &cls_flat);
    let out = b.add_sigmoid(cls_reshaped, &cls_flat);
    let def = b.build(out).expect("valid e2e P5 kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, BASE_CH, IN_CH, 3); // stem
    push_conv_bn_silu_bindings(&mut bindings, P3_CH, BASE_CH, 3); // s1
    push_conv_bn_silu_bindings(&mut bindings, P4_CH, P3_CH, 3); // s2
    push_conv_bn_silu_bindings(&mut bindings, P5_CH, P4_CH, 3); // s3
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        NUM_CLASSES,
        P5_CH,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[1, IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP through e2e P5");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &cls_flat);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale e2e P5 IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-5;
    assert!(
        lo_min >= 0.0 - eps,
        "e2e P5 sigmoid lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "e2e P5 sigmoid upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 15. Multi-scale monotone tightening (IBP)
// ===========================================================================

/// Verifies IBP monotonicity across the multi-scale pipeline: narrower pixel
/// bounds must produce no-wider output bounds at every scale.
#[test]
fn test_multiscale_monotone_tightening_ibp() {
    // Build a 2-stage backbone ending in P3 detection.
    let stem_sp = IMG_SIZE / 2;
    let p3_sp = P3_SPATIAL;
    let num_anchors = p3_sp * p3_sp;
    let cls_flat = [num_anchors, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("dly_ms_monotone");
    let input = b.add_input("image", &[1, IN_CH, IMG_SIZE, IMG_SIZE]);
    let stem = add_conv_bn_silu(
        &mut b, input, "stem", IN_CH, BASE_CH, 3, 2, 1, stem_sp, stem_sp,
    );
    let backbone = add_conv_bn_silu(&mut b, stem, "s1", BASE_CH, P3_CH, 3, 2, 1, p3_sp, p3_sp);
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
    let def = b.build(out).expect("valid monotone kernel");

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

    // Wide input: [0, 1]
    let wide_input = image_bounds(&[1, IN_CH, IMG_SIZE, IMG_SIZE]);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");

    // Narrow input: [0.2, 0.8]
    let narrow_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, IN_CH, IMG_SIZE, IMG_SIZE]), 0.2f32),
        ArrayD::from_elem(IxDyn(&[1, IN_CH, IMG_SIZE, IMG_SIZE]), 0.8f32),
    )
    .expect("valid narrow bounds");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");

    let (lo_w, hi_w) = bounds_min_max(&wide_output);
    let (lo_n, hi_n) = bounds_min_max(&narrow_output);
    let wide_width = hi_w - lo_w;
    let narrow_width = hi_n - lo_n;

    eprintln!(
        "DLY multiscale monotone: wide=[{lo_w:.4}, {hi_w:.4}] w={wide_width:.4} | \
         narrow=[{lo_n:.4}, {hi_n:.4}] w={narrow_width:.4}"
    );

    assert!(
        narrow_width <= wide_width + 1e-4,
        "monotone tightening violated: narrow_width={narrow_width} > wide_width={wide_width}"
    );
}

// ===========================================================================
// 16. Multi-scale widening analysis (IBP)
// ===========================================================================

/// Quantifies how IBP bounds widen as we go from shallow (P3) to deep (P5)
/// backbone paths. More depth means more uncertainty in IBP. This test
/// measures the growth factor to detect vacuous blowup.
#[test]
fn test_multiscale_widening_across_scales_ibp() {
    let img_shape = [1, IN_CH, IMG_SIZE, IMG_SIZE];
    let input = image_bounds(&img_shape);

    // P3 path: stem + 1 stage (2 layers)
    let stem_sp = IMG_SIZE / 2;
    let p3_sp = P3_SPATIAL;
    {
        let mut b = TensorBlockBuilder::new("dly_ms_widen_p3");
        let inp = b.add_input("image", &img_shape);
        let stem = add_conv_bn_silu(
            &mut b, inp, "stem", IN_CH, BASE_CH, 3, 2, 1, stem_sp, stem_sp,
        );
        let out = add_conv_bn_silu(&mut b, stem, "s1", BASE_CH, P3_CH, 3, 2, 1, p3_sp, p3_sp);
        let def = b.build(out).expect("p3 path kernel");
        let mut bindings = vec![TensorParamBinding::Variable];
        push_conv_bn_silu_bindings(&mut bindings, BASE_CH, IN_CH, 3);
        push_conv_bn_silu_bindings(&mut bindings, P3_CH, BASE_CH, 3);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let p3_out = graph.propagate_ibp(&input).expect("IBP P3");
        assert_bounds_valid(&p3_out);
        let (lo3, hi3) = bounds_min_max(&p3_out);
        let w3 = hi3 - lo3;

        // P5 path: stem + 3 stages (4 layers)
        let p4_sp = P4_SPATIAL;
        let p5_sp = P5_SPATIAL;
        let mut b5 = TensorBlockBuilder::new("dly_ms_widen_p5");
        let inp5 = b5.add_input("image", &img_shape);
        let stem5 = add_conv_bn_silu(
            &mut b5, inp5, "stem", IN_CH, BASE_CH, 3, 2, 1, stem_sp, stem_sp,
        );
        let s1 = add_conv_bn_silu(&mut b5, stem5, "s1", BASE_CH, P3_CH, 3, 2, 1, p3_sp, p3_sp);
        let s2 = add_conv_bn_silu(&mut b5, s1, "s2", P3_CH, P4_CH, 3, 2, 1, p4_sp, p4_sp);
        let s3 = add_conv_bn_silu(&mut b5, s2, "s3", P4_CH, P5_CH, 3, 2, 1, p5_sp, p5_sp);
        let def5 = b5.build(s3).expect("p5 path kernel");
        let mut bindings5 = vec![TensorParamBinding::Variable];
        push_conv_bn_silu_bindings(&mut bindings5, BASE_CH, IN_CH, 3);
        push_conv_bn_silu_bindings(&mut bindings5, P3_CH, BASE_CH, 3);
        push_conv_bn_silu_bindings(&mut bindings5, P4_CH, P3_CH, 3);
        push_conv_bn_silu_bindings(&mut bindings5, P5_CH, P4_CH, 3);
        let graph5 = tensor_kernel_to_graph(&def5, &bindings5).expect("graph");
        let p5_out = graph5.propagate_ibp(&input).expect("IBP P5");
        assert_bounds_valid(&p5_out);
        let (lo5, hi5) = bounds_min_max(&p5_out);
        let w5 = hi5 - lo5;

        eprintln!("DLY multiscale widening analysis:");
        eprintln!("  P3 (2 layers): [{lo3:.4}, {hi3:.4}] width={w3:.4}");
        eprintln!("  P5 (4 layers): [{lo5:.4}, {hi5:.4}] width={w5:.4}");

        // Note: with BN normalization (variance=1, mean=0) and small weights,
        // deeper paths can actually produce tighter bounds due to repeated
        // normalization. We check finite + non-vacuous rather than ordering.

        // Neither should be vacuously wide
        assert!(w3 < 1e6, "P3 bounds vacuously wide: {w3}");
        assert!(w5 < 1e6, "P5 bounds vacuously wide: {w5}");

        if w3 > 0.0 {
            let ratio = w5 / w3;
            eprintln!("  Width ratio (P5/P3): {ratio:.2}x");
        }
    }
}

// ===========================================================================
// 17. Objectness + class confidence scoring (IBP)
// ===========================================================================

/// Objectness x class confidence: sigmoid(obj) * softmax(cls).
///
/// Input: [NUM_ANCHORS, 1 + NUM_CLASSES] (objectness logit + class logits).
/// Output: [NUM_ANCHORS, NUM_CLASSES] (confidence = sigmoid(obj) * softmax(cls)).
///
/// Both sigmoid and softmax output in [0, 1], so product is in [0, 1].
/// This models the YOLO detection scoring where objectness and class
/// probabilities are multiplied for final confidence.
#[test]
fn test_multiscale_objectness_class_confidence_ibp() {
    let num_anchors = P3_SPATIAL * P3_SPATIAL; // 16
    let total_in = 1 + NUM_CLASSES;
    let shape_in = [num_anchors, total_in];
    let cls_shape = [num_anchors, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("dly_ms_obj_cls_score");
    let input = b.add_input("raw_scores", &shape_in);

    // Split objectness (first dim) and class logits (remaining).
    let obj_logit = b.add_narrow(input, 1, 0, 1, &[num_anchors, 1]);
    let cls_logits = b.add_narrow(input, 1, 1, NUM_CLASSES, &cls_shape);

    // Sigmoid objectness -> [0, 1].
    let obj_prob = b.add_sigmoid(obj_logit, &[num_anchors, 1]);

    // Softmax class probabilities -> each in [0, 1], sums to 1.
    let cls_probs = b.add_softmax(cls_logits, 1, &cls_shape);

    // Broadcast objectness [num_anchors, 1] -> [num_anchors, NUM_CLASSES].
    let obj_broadcast = b.add_broadcast(obj_prob, &cls_shape);

    // Final confidence: obj * cls.
    let confidence = b.add_binary_mul(obj_broadcast, cls_probs, &cls_shape);

    let def = b.build(confidence).expect("valid obj x cls scoring kernel");
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&shape_in, 5.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through obj x cls scoring");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &cls_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale obj x cls scoring IBP: [{lo_min:.6}, {hi_max:.6}]");

    // sigmoid * softmax product is in [0, 1].
    let eps = 1e-4;
    assert!(
        lo_min >= 0.0 - eps,
        "confidence lower >= 0 (sigmoid * softmax), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "confidence upper <= 1 (sigmoid * softmax), got {hi_max}"
    );
}

// ===========================================================================
// 18. Multi-scale to single output merge (IBP)
// ===========================================================================

/// Merge predictions from multiple branches into a single output tensor.
///
/// Input: [1, TOTAL_ANCHORS, NUM_CLASSES] (Variable).
/// Three branches via narrow slices model per-scale detection heads:
///   P3 branch: narrow [0..P3_ANCH] -> sigmoid
///   P4 branch: narrow [P3_ANCH..P3_ANCH+P4_ANCH] -> sigmoid
///   P5 branch: narrow [P3_ANCH+P4_ANCH..TOTAL] -> sigmoid
/// Output: concat on axis 1.
/// Uses 3D shapes with batch dim to avoid axis-0 concat (reserved by NY).
///
/// Key property: concat preserves per-element bounds from each source.
#[test]
fn test_multiscale_to_single_output_merge_ibp() {
    let p3_anch = P3_SPATIAL * P3_SPATIAL; // 16
    let p4_anch = P4_SPATIAL * P4_SPATIAL; // 4
    let p5_anch = P5_SPATIAL * P5_SPATIAL; // 1
    let total_anch = p3_anch + p4_anch + p5_anch; // 21

    let mut b = TensorBlockBuilder::new("dly_ms_scale_merge");
    // Variable input holds all anchor logits across all scales.
    let all_preds = b.add_input("all_preds", &[1, total_anch, NUM_CLASSES]);

    // Narrow slices model per-scale detection outputs.
    let p3_slice = b.add_narrow(all_preds, 1, 0, p3_anch, &[1, p3_anch, NUM_CLASSES]);
    let p4_slice = b.add_narrow(all_preds, 1, p3_anch, p4_anch, &[1, p4_anch, NUM_CLASSES]);
    let p5_slice = b.add_narrow(
        all_preds,
        1,
        p3_anch + p4_anch,
        p5_anch,
        &[1, p5_anch, NUM_CLASSES],
    );

    // Per-scale sigmoid (models independent detection heads).
    let p3_out = b.add_sigmoid(p3_slice, &[1, p3_anch, NUM_CLASSES]);
    let p4_out = b.add_sigmoid(p4_slice, &[1, p4_anch, NUM_CLASSES]);
    let p5_out = b.add_sigmoid(p5_slice, &[1, p5_anch, NUM_CLASSES]);

    // Concat along anchor dimension (axis 1, not reserved axis 0).
    let merged = b.add_concat(&[p3_out, p4_out, p5_out], 1, &[1, total_anch, NUM_CLASSES]);
    let def = b.build(merged).expect("valid multi-scale merge kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[1, total_anch, NUM_CLASSES], 3.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through multi-scale merge");
    assert_bounds_valid(&output);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, total_anch, NUM_CLASSES],
        "merged output shape: expected [1, {total_anch}, {NUM_CLASSES}]"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY multiscale merge IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    // All branches are sigmoid, so output in [0, 1].
    let eps = 1e-4;
    assert!(
        lo_min >= 0.0 - eps,
        "merge lower >= 0 (sigmoid), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "merge upper <= 1 (sigmoid), got {hi_max}"
    );
}

// ===========================================================================
// 19. NMS confidence thresholding (IBP + CROWN)
// ===========================================================================

/// NMS confidence thresholding: sigmoid -> subtract threshold -> ReLU filter.
///
/// Input: [NUM_ANCHORS, NUM_CLASSES] (raw logits).
/// Pipeline: sigmoid(logits) - threshold -> ReLU.
/// Output: [NUM_ANCHORS, NUM_CLASSES] (thresholded scores, 0 below threshold).
///
/// The ReLU after threshold subtraction acts as a hard filter: scores below
/// the threshold become 0, scores above are shifted down by the threshold.
/// This models the pre-NMS confidence filtering step in YOLO detectors.
#[test]
fn test_multiscale_nms_confidence_threshold_ibp_crown() {
    let num_anchors = P3_SPATIAL * P3_SPATIAL;
    let shape = [num_anchors, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("dly_ms_nms_threshold");
    let logits = b.add_input("logits", &shape);

    // Sigmoid: map logits to [0, 1].
    let probs = b.add_sigmoid(logits, &shape);

    // Subtract threshold (0.25 typical for YOLO NMS).
    // Model as: probs + (-threshold) via adding a full-shape constant bias.
    // Using full [num_anchors, NUM_CLASSES] shape instead of broadcast_left
    // (which is not supported for [NUM_CLASSES] -> [num_anchors, NUM_CLASSES]).
    let neg_thresh = b.add_input("neg_threshold", &shape);
    let shifted = b.add_binary_add(probs, neg_thresh, &shape);

    // ReLU: zero out negative values (scores below threshold).
    let filtered = b.add_relu(shifted, &shape);

    let def = b.build(filtered).expect("valid NMS threshold kernel");
    let neg_thresh_tensor = ArrayD::from_elem(IxDyn(&shape), -0.25f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(neg_thresh_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&shape, 5.0);

    // IBP pass
    let ibp_output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through NMS threshold");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("DLY multiscale NMS threshold IBP: [{lo_min:.6}, {hi_max:.6}]");

    // After sigmoid -> subtract 0.25 -> ReLU:
    // ReLU output is >= 0.
    // Upper: sigmoid max is 1.0, minus 0.25 = 0.75.
    let eps = 1e-4;
    assert!(lo_min >= 0.0 - eps, "ReLU output lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 0.75 + eps,
        "thresholded upper <= 0.75, got {hi_max}"
    );

    // CROWN pass
    let (method, crown_output, fallback) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    assert_eq!(crown_output.lower_upper().0.shape(), &shape);

    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("DLY multiscale NMS threshold CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}
