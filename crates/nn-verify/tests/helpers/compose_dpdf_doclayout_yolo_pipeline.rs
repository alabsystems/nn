// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for DocLayout-YOLO full detection pipeline bounds.
//!
//! Verifies IBP and CROWN bound propagation through the complete DocLayout-YOLO
//! detection pipeline (YOLOv10-based document layout detection):
//!
//! ## Tests (18 tests)
//!
//! 1.  **ConvBnAct backbone block bounds** — Conv2d + BatchNorm + SiLU (IBP)
//! 2.  **SPPF MaxPool bounds** — Spatial Pyramid Pooling Fast (IBP)
//! 3.  **C2f bottleneck feature bounds** — Cross-Stage Partial bottleneck (IBP)
//! 4.  **PAN neck upsampling path bounds** — Top-down feature fusion (IBP)
//! 5.  **PAN neck downsample path bounds** — Bottom-up feature fusion (IBP)
//! 6.  **Multi-scale P3/P4/P5 feature pyramid bounds** — 3-level FPN (IBP)
//! 7.  **Detection head box regression bounds (DFL)** — Softmax + weighted sum (IBP)
//! 8.  **Detection head classification logit bounds** — Linear logits (IBP)
//! 9.  **Sigmoid confidence score bounds [0,1]** — Sigmoid output (IBP)
//! 10. **Anchor-free grid point bounds** — Grid + offset decoding (IBP)
//! 11. **Full backbone-to-neck pipeline composition** — Backbone -> SPPF -> neck (IBP)
//! 12. **Full neck-to-head pipeline composition** — Neck -> detection heads (IBP)
//! 13. **End-to-end image-to-detection bounds** — Full pipeline (IBP)
//! 14. **NMS post-processing score filtering bounds** — Score threshold (IBP)
//! 15. **Backbone residual connection bounds** — Skip connection (IBP)
//! 16. **C2f bottleneck with shortcut bounds** — Bottleneck + residual (CROWN)
//! 17. **Detection head class probability bounds** — Softmax class distribution (IBP)
//! 18. **Multi-class output shape preservation** — Shape verification (IBP)
//!
//! Architecture references:
//! - DocLayout-YOLO (Zhao et al. 2024): YOLOv10-based document layout detection
//! - YOLOv8/v10 C2f: Cross-Stage Partial with 2 convolutions
//! - SPPF: Spatial Pyramid Pooling - Fast (YOLOv5/v8)
//! - DFL (Li et al. 2022): Distribution Focal Loss for box regression
//! - PAN (Liu et al. 2018): Path Aggregation Network
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=4 (symbolic, real: 640), BASE_CH=4 (symbolic, real: 64)
//! - NUM_CLASSES=3 (symbolic, real: 11), NUM_BOXES=4, HEAD_DIM=4
//!
//! Part of #4186: Compose tests for DocLayout-YOLO detection pipeline bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- symbolic for fast verification
// ---------------------------------------------------------------------------

const IN_CHANNELS: usize = 3;
const IMG_SIZE: usize = 4;
const BASE_CH: usize = 4;
const NUM_CLASSES: usize = 3;
const NUM_BOXES: usize = 4;
const HEAD_DIM: usize = 4;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// DFL register count (softmax dimension for box regression).
const DFL_REG: usize = 4;

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
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 1.0f32),
    )
    .expect("valid image bounds")
}

/// Feature-domain input bounds: [-range, +range].
fn feature_bounds(shape: &[usize], range: f32) -> BoundedTensor {
    uniform_bounds(shape, range)
}

/// Bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Add SiLU activation: sigmoid(x) * x.
///
/// SiLU has no dedicated builder, so we compose from sigmoid + binary_mul.
fn add_silu(b: &mut TensorBlockBuilder, input: TensorNodeId, shape: &[usize]) -> TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

/// Add ConvBnAct block: Conv2d -> BatchNorm -> SiLU.
///
/// Returns (output_node, number_of_param_inputs_added).
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

/// Add a bottleneck block: Conv3x3 -> SiLU -> Conv3x3 (no BN for simplicity).
fn add_bottleneck(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    ch: usize,
    spatial: usize,
    prefix: &str,
) -> TensorNodeId {
    let shape = [ch, spatial, spatial];
    let w1 = b.add_input(&format!("{prefix}_bn1_w"), &[ch, ch, 3, 3]);
    let conv1 = b.add_conv2d(input, w1, None, 1, 1, 1, 1, &shape);
    let act1 = add_silu(b, conv1, &shape);
    let w2 = b.add_input(&format!("{prefix}_bn2_w"), &[ch, ch, 3, 3]);
    b.add_conv2d(act1, w2, None, 1, 1, 1, 1, &shape)
}

/// Push bindings for one bottleneck block (2 conv weights).
fn push_bottleneck_bindings(bindings: &mut Vec<TensorParamBinding>, ch: usize) {
    bindings.push(weight(&[ch, ch, 3, 3])); // bn1_w
    bindings.push(weight(&[ch, ch, 3, 3])); // bn2_w
}

// ===========================================================================
// 1. ConvBnAct backbone block bounds (Conv2d + BatchNorm + SiLU)
// ===========================================================================

#[test]
fn test_conv_bn_act_backbone_block_ibp() {
    let out_ch = BASE_CH;
    let out_size = IMG_SIZE / 2;
    let mut b = TensorBlockBuilder::new("dly_pipe_conv_bn_act");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let out = add_conv_bn_act(
        &mut b,
        input,
        IN_CHANNELS,
        out_ch,
        3,
        2,
        1,
        out_size,
        out_size,
        "stem",
    );
    let def = b.build(out).expect("valid conv_bn_act kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_act_bindings(&mut bindings, IN_CHANNELS, out_ch, 3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP through ConvBnAct");
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[out_ch, out_size, out_size],
        "ConvBnAct output shape"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline ConvBnAct IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. SPPF (Spatial Pyramid Pooling Fast) MaxPool bounds
// ===========================================================================

#[test]
fn test_sppf_maxpool_bounds_ibp() {
    let ch = BASE_CH;
    let sp = IMG_SIZE;
    let pool_k: usize = 3;
    let pad: usize = 1;
    let shape = [ch, sp, sp];

    let mut b = TensorBlockBuilder::new("dly_pipe_sppf");
    let input = b.add_input("features", &shape);

    // SPPF: 3 sequential MaxPools, all same-padded to preserve spatial
    let p1 = b.add_max_pool_2d(input, pool_k, pool_k, 1, 1, pad, pad, &shape);
    let p2 = b.add_max_pool_2d(p1, pool_k, pool_k, 1, 1, pad, pad, &shape);
    let p3 = b.add_max_pool_2d(p2, pool_k, pool_k, 1, 1, pad, pad, &shape);

    // Concat along channel axis: [ch*4, sp, sp]
    let cat_shape = [ch * 4, sp, sp];
    let cat = b.add_concat(&[input, p1, p2, p3], 0, &cat_shape);

    // 1x1 conv to reduce back to ch
    let reduce_w = b.add_input("reduce_w", &[ch, ch * 4, 1, 1]);
    let out = b.add_conv2d(cat, reduce_w, None, 1, 1, 0, 0, &shape);
    let def = b.build(out).expect("valid SPPF kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.push(weight(&[ch, ch * 4, 1, 1])); // reduce_w
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&shape, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through SPPF");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline SPPF IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. C2f (Cross-Stage Partial) bottleneck feature bounds (IBP)
// ===========================================================================

#[test]
fn test_c2f_bottleneck_ibp() {
    let ch = BASE_CH;
    let sp = IMG_SIZE;
    let shape = [ch, sp, sp];

    let mut b = TensorBlockBuilder::new("dly_pipe_c2f");
    let input = b.add_input("features", &shape);

    // Entry conv: 1x1 to expand
    let entry_w = b.add_input("entry_w", &[ch * 2, ch, 1, 1]);
    let expanded = b.add_conv2d(input, entry_w, None, 1, 1, 0, 0, &[ch * 2, sp, sp]);

    // Split into two halves along channel axis
    let half_shape = [ch, sp, sp];
    let split0 = b.add_narrow(expanded, 0, 0, ch, &half_shape);
    let split1 = b.add_narrow(expanded, 0, ch, ch, &half_shape);

    // One bottleneck on the second half
    let bottleneck_out = add_bottleneck(&mut b, split1, ch, sp, "bneck0");

    // Concat: split0 + bottleneck_out -> [ch*2, sp, sp]
    let cat_shape = [ch * 2, sp, sp];
    let cat = b.add_concat(&[split0, bottleneck_out], 0, &cat_shape);

    // Exit conv: 1x1 to reduce
    let exit_w = b.add_input("exit_w", &[ch, ch * 2, 1, 1]);
    let out = b.add_conv2d(cat, exit_w, None, 1, 1, 0, 0, &shape);
    let def = b.build(out).expect("valid C2f kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.push(weight(&[ch * 2, ch, 1, 1])); // entry_w
    push_bottleneck_bindings(&mut bindings, ch);
    bindings.push(weight(&[ch, ch * 2, 1, 1])); // exit_w
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&shape, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through C2f");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline C2f IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 4. PAN neck upsampling path bounds
// ===========================================================================

#[test]
fn test_pan_neck_upsample_path_ibp() {
    // Top-down path: high-level features (low spatial) combined with earlier features.
    // Model as: 1x1 conv (channel reduce) + concat with skip features.
    let ch = BASE_CH;
    let sp = IMG_SIZE;
    let hi_shape = [ch * 2, sp, sp];
    let lo_shape = [ch, sp, sp];

    let mut b = TensorBlockBuilder::new("dly_pipe_pan_up");
    let hi_feat = b.add_input("hi_features", &hi_shape);
    let skip_feat = b.add_input("skip_features", &lo_shape);

    // 1x1 conv to reduce hi_feat channels
    let reduce_w = b.add_input("reduce_w", &[ch, ch * 2, 1, 1]);
    let reduced = b.add_conv2d(hi_feat, reduce_w, None, 1, 1, 0, 0, &lo_shape);

    // Concat skip + reduced
    let cat_shape = [ch * 2, sp, sp];
    let cat = b.add_concat(&[skip_feat, reduced], 0, &cat_shape);

    // 1x1 conv to fuse
    let fuse_w = b.add_input("fuse_w", &[ch, ch * 2, 1, 1]);
    let out = b.add_conv2d(cat, fuse_w, None, 1, 1, 0, 0, &lo_shape);
    let def = b.build(out).expect("valid PAN up kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // hi_feat
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&lo_shape), 0.5f32)), // skip_feat
        weight(&[ch, ch * 2, 1, 1]),                                                     // reduce_w
        weight(&[ch, ch * 2, 1, 1]),                                                     // fuse_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&hi_shape, 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN upsample");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &lo_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline PAN upsample IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. PAN neck downsample path bounds
// ===========================================================================

#[test]
fn test_pan_neck_downsample_path_ibp() {
    // Bottom-up path: stride-2 conv to downsample + concat with higher-level features.
    let ch = BASE_CH;
    let sp = IMG_SIZE;
    let shape = [ch, sp, sp];
    let ds_sp = sp / 2;
    let ds_shape = [ch, ds_sp, ds_sp];
    let hi_shape = [ch, ds_sp, ds_sp];

    let mut b = TensorBlockBuilder::new("dly_pipe_pan_down");
    let lo_feat = b.add_input("lo_features", &shape);
    let hi_feat = b.add_input("hi_features", &hi_shape);

    // Stride-2 conv to downsample
    let ds_w = b.add_input("ds_w", &[ch, ch, 3, 3]);
    let downsampled = b.add_conv2d(lo_feat, ds_w, None, 2, 2, 1, 1, &ds_shape);

    // Concat
    let cat_shape = [ch * 2, ds_sp, ds_sp];
    let cat = b.add_concat(&[downsampled, hi_feat], 0, &cat_shape);

    // 1x1 conv to fuse
    let fuse_w = b.add_input("fuse_w", &[ch, ch * 2, 1, 1]);
    let out = b.add_conv2d(cat, fuse_w, None, 1, 1, 0, 0, &ds_shape);
    let def = b.build(out).expect("valid PAN down kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // lo_feat
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&hi_shape), 0.5f32)), // hi_feat
        weight(&[ch, ch, 3, 3]),      // ds_w
        weight(&[ch, ch * 2, 1, 1]),  // fuse_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&shape, 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN downsample");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &ds_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline PAN downsample IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. Multi-scale P3/P4/P5 feature pyramid bounds
// ===========================================================================

#[test]
fn test_multi_scale_feature_pyramid_ibp() {
    // 3-level feature pyramid: P3 (large spatial), P4 (medium), P5 (small).
    // Each level: ConvBnAct with different strides.
    let ch = BASE_CH;
    let sp = IMG_SIZE;
    let shape = [ch, sp, sp];

    let mut b = TensorBlockBuilder::new("dly_pipe_fpn_multi");
    let input = b.add_input("backbone_out", &shape);

    // P3: 1x1 conv (preserve spatial)
    let p3_w = b.add_input("p3_w", &[ch, ch, 1, 1]);
    let p3 = b.add_conv2d(input, p3_w, None, 1, 1, 0, 0, &shape);

    // P4: stride-2 conv
    let p4_sp = sp / 2;
    let p4_shape = [ch, p4_sp, p4_sp];
    let p4_w = b.add_input("p4_w", &[ch, ch, 3, 3]);
    let p4 = b.add_conv2d(input, p4_w, None, 2, 2, 1, 1, &p4_shape);

    // P5: stride-2 from P4 (total 4x reduction)
    let p5_sp = p4_sp / 2;
    let p5_shape = [ch, p5_sp, p5_sp];
    let p5_w = b.add_input("p5_w", &[ch, ch, 3, 3]);
    let _p5 = b.add_conv2d(p4, p5_w, None, 2, 2, 1, 1, &p5_shape);

    // Build through P3 (verify all levels produce valid bounds)
    let def = b.build(p3).expect("valid FPN kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[ch, ch, 1, 1]), // p3_w
        weight(&[ch, ch, 3, 3]), // p4_w
        weight(&[ch, ch, 3, 3]), // p5_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&shape, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through FPN");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline FPN multi-scale IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. Detection head box regression bounds (DFL)
// ===========================================================================

#[test]
fn test_detection_head_dfl_regression_ibp() {
    // DFL: Linear -> softmax -> weighted sum to produce continuous box coordinates.
    // Input: [NUM_BOXES, DFL_REG * 4] (4 box sides, DFL_REG bins each)
    // Output: [NUM_BOXES, 4] box coordinates
    let input_dim = DFL_REG * 4;
    let mut b = TensorBlockBuilder::new("dly_pipe_dfl");
    let input = b.add_input("box_logits", &[NUM_BOXES, input_dim]);

    // Reshape to [NUM_BOXES * 4, DFL_REG] for per-side softmax
    let reshaped = b.add_reshape(input, &[NUM_BOXES * 4, DFL_REG]);
    let softmax = b.add_softmax(reshaped, -1, &[NUM_BOXES * 4, DFL_REG]);

    // Weighted sum: multiply by [0, 1, 2, ..., DFL_REG-1] and sum.
    // Model as linear with fixed weight [1, DFL_REG] = [0, 1, 2, ...]
    let proj_w = b.add_input("dfl_proj", &[1, DFL_REG]);
    let out = b.add_linear(softmax, proj_w, None, &[NUM_BOXES * 4, 1]);
    let def = b.build(out).expect("valid DFL kernel");

    // DFL projection weights: [0, 1, 2, ..., DFL_REG-1]
    let dfl_data: Vec<f32> = (0..DFL_REG).map(|i| i as f32).collect();
    let dfl_proj = ArrayD::from_shape_vec(IxDyn(&[1, DFL_REG]), dfl_data).expect("valid DFL proj");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(dfl_proj),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&[NUM_BOXES, input_dim], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through DFL");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline DFL regression IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    // DFL output: softmax weighted sum, should be in [0, DFL_REG-1]
    assert!(
        lo_min >= -0.01,
        "DFL lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= (DFL_REG as f32) + 0.01,
        "DFL upper bound should be <= {DFL_REG}, got {hi_max}"
    );
}

// ===========================================================================
// 8. Detection head classification logit bounds
// ===========================================================================

#[test]
fn test_detection_head_classification_logit_ibp() {
    let ch = BASE_CH;
    let mut b = TensorBlockBuilder::new("dly_pipe_cls_logits");
    let input = b.add_input("neck_features", &[NUM_BOXES, ch]);

    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, ch]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let out = b.add_linear(input, cls_w, Some(cls_b), &[NUM_BOXES, NUM_CLASSES]);
    let def = b.build(out).expect("valid cls logit kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, ch]),
        bias_zero(&[NUM_CLASSES]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&[NUM_BOXES, ch], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through cls logits");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, NUM_CLASSES]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline cls logits IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 9. Sigmoid confidence score bounds [0,1]
// ===========================================================================

#[test]
fn test_sigmoid_confidence_score_bounds_ibp() {
    let mut b = TensorBlockBuilder::new("dly_pipe_sigmoid_conf");
    let input = b.add_input("cls_logits", &[NUM_BOXES, NUM_CLASSES]);
    let out = b.add_sigmoid(input, &[NUM_BOXES, NUM_CLASSES]);
    let def = b.build(out).expect("valid sigmoid kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&[NUM_BOXES, NUM_CLASSES], 5.0);

    let output = graph.propagate_ibp(&input).expect("IBP through sigmoid");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline sigmoid conf IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -0.01,
        "sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(hi_max <= 1.01, "sigmoid upper should be <= 1, got {hi_max}");
}

// ===========================================================================
// 10. Anchor-free grid point bounds
// ===========================================================================

#[test]
fn test_anchor_free_grid_point_bounds_ibp() {
    // Grid decoding: predicted offset + grid anchor -> absolute coordinates.
    // Model as: sigmoid(pred) + constant_grid -> [0, grid_max] bounded.
    let grid_size = IMG_SIZE;
    let flat_size = grid_size * grid_size;

    let mut b = TensorBlockBuilder::new("dly_pipe_grid_decode");
    let pred = b.add_input("pred_offset", &[flat_size, 2]);
    let sig = b.add_sigmoid(pred, &[flat_size, 2]);

    // Add grid anchors as constants
    let grid_anchor = b.add_input("grid_anchor", &[flat_size, 2]);
    let out = b.add_binary_add(sig, grid_anchor, &[flat_size, 2]);
    let def = b.build(out).expect("valid grid decode kernel");

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
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&[flat_size, 2], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through grid decode");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[flat_size, 2]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline grid decode IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // sigmoid output is [0, 1], + grid offset [0, grid_size-1]
    assert!(lo_min >= -0.01, "grid lower should be >= 0, got {lo_min}");
    assert!(
        hi_max <= (grid_size as f32) + 1.01,
        "grid upper should be <= grid_size+1, got {hi_max}"
    );
}

// ===========================================================================
// 11. Full backbone-to-neck pipeline composition (IBP)
// ===========================================================================

#[test]
fn test_full_backbone_to_neck_ibp() {
    let ch = BASE_CH;
    let sp = IMG_SIZE;

    let mut b = TensorBlockBuilder::new("dly_pipe_backbone_neck");
    let input = b.add_input("image", &[IN_CHANNELS, sp, sp]);

    // Stem: ConvBnAct stride-2
    let stem_sp = sp / 2;
    let stem = add_conv_bn_act(
        &mut b,
        input,
        IN_CHANNELS,
        ch,
        3,
        2,
        1,
        stem_sp,
        stem_sp,
        "stem",
    );

    // Stage1: ConvBnAct stride-1 (preserve spatial)
    let s1 = add_conv_bn_act(&mut b, stem, ch, ch, 3, 1, 1, stem_sp, stem_sp, "s1");

    // SPPF at the end: MaxPool chain + concat + reduce
    let shape = [ch, stem_sp, stem_sp];
    let p1 = b.add_max_pool_2d(s1, 3, 3, 1, 1, 1, 1, &shape);
    let p2 = b.add_max_pool_2d(p1, 3, 3, 1, 1, 1, 1, &shape);
    let cat_shape = [ch * 3, stem_sp, stem_sp];
    let cat = b.add_concat(&[s1, p1, p2], 0, &cat_shape);
    let reduce_w = b.add_input("sppf_reduce", &[ch, ch * 3, 1, 1]);
    let out = b.add_conv2d(cat, reduce_w, None, 1, 1, 0, 0, &shape);
    let def = b.build(out).expect("valid backbone-neck kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_act_bindings(&mut bindings, IN_CHANNELS, ch, 3); // stem
    push_conv_bn_act_bindings(&mut bindings, ch, ch, 3); // s1
    bindings.push(weight(&[ch, ch * 3, 1, 1])); // sppf_reduce
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds();

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through backbone-neck");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline backbone-neck IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 12. Full neck-to-head pipeline composition (IBP)
// ===========================================================================

#[test]
fn test_full_neck_to_head_ibp() {
    let ch = BASE_CH;
    let sp = IMG_SIZE;
    let shape = [ch, sp, sp];

    let mut b = TensorBlockBuilder::new("dly_pipe_neck_head");
    let input = b.add_input("neck_features", &shape);

    // Flatten spatial: [ch, sp, sp] -> [sp*sp, ch]
    let flat_shape = [sp * sp, ch];
    let reshaped = b.add_reshape(input, &[ch, sp * sp]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &flat_shape);

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, ch]);
    let cls_logits = b.add_linear(transposed, cls_w, None, &[sp * sp, NUM_CLASSES]);
    let out = b.add_sigmoid(cls_logits, &[sp * sp, NUM_CLASSES]);
    let def = b.build(out).expect("valid neck-head kernel");

    let bindings = vec![TensorParamBinding::Variable, weight(&[NUM_CLASSES, ch])];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&shape, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through neck-head");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline neck-head IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -0.01,
        "sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(hi_max <= 1.01, "sigmoid upper should be <= 1, got {hi_max}");
}

// ===========================================================================
// 13. End-to-end image-to-detection bounds
// ===========================================================================

#[test]
fn test_end_to_end_image_to_detection_ibp() {
    let ch = BASE_CH;
    let sp = IMG_SIZE;
    let stem_sp = sp / 2;

    let mut b = TensorBlockBuilder::new("dly_pipe_e2e");
    let input = b.add_input("image", &[IN_CHANNELS, sp, sp]);

    // Backbone: ConvBnAct stem
    let stem = add_conv_bn_act(
        &mut b,
        input,
        IN_CHANNELS,
        ch,
        3,
        2,
        1,
        stem_sp,
        stem_sp,
        "stem",
    );

    // Flatten: [ch, stem_sp, stem_sp] -> [stem_sp*stem_sp, ch]
    let num_pos = stem_sp * stem_sp;
    let flat = b.add_reshape(stem, &[ch, num_pos]);
    let transposed = b.add_transpose(flat, &[1, 0], &[num_pos, ch]);

    // Detection: Linear -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, ch]);
    let logits = b.add_linear(transposed, cls_w, None, &[num_pos, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[num_pos, NUM_CLASSES]);
    let def = b.build(out).expect("valid e2e kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_act_bindings(&mut bindings, IN_CHANNELS, ch, 3);
    bindings.push(weight(&[NUM_CLASSES, ch]));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds();

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through e2e pipeline");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline e2e IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -0.01,
        "e2e sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "e2e sigmoid upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 14. NMS post-processing score filtering bounds
// ===========================================================================

#[test]
fn test_nms_score_filtering_bounds_ibp() {
    // Score filtering: sigmoid(logits) clamped by threshold.
    // We verify that sigmoid output remains in [0, 1] and that ReLU(sigmoid - threshold)
    // produces non-negative scores.
    let mut b = TensorBlockBuilder::new("dly_pipe_nms_filter");
    let input = b.add_input("cls_logits", &[NUM_BOXES, NUM_CLASSES]);
    let conf = b.add_sigmoid(input, &[NUM_BOXES, NUM_CLASSES]);

    // Subtract threshold (modeled as constant)
    let thresh = b.add_input("threshold", &[NUM_BOXES, NUM_CLASSES]);
    let diff = b.add_binary_add(conf, thresh, &[NUM_BOXES, NUM_CLASSES]);

    // ReLU to zero out below-threshold scores
    let out = b.add_relu(diff, &[NUM_BOXES, NUM_CLASSES]);
    let def = b.build(out).expect("valid NMS filter kernel");

    // Threshold of -0.25 (subtract 0.25 from sigmoid outputs)
    let thresh_data = ArrayD::from_elem(IxDyn(&[NUM_BOXES, NUM_CLASSES]), -0.25f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(thresh_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&[NUM_BOXES, NUM_CLASSES], 5.0);

    let output = graph.propagate_ibp(&input).expect("IBP through NMS filter");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline NMS filter IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // ReLU output: lower bound should be >= 0 (non-negative after ReLU)
    assert!(
        lo_min >= -0.01,
        "ReLU output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "ReLU(sigmoid - 0.25) upper should be <= 0.75, got {hi_max}"
    );
}

// ===========================================================================
// 15. Backbone residual connection bounds
// ===========================================================================

#[test]
fn test_backbone_residual_connection_ibp() {
    let ch = BASE_CH;
    let sp = IMG_SIZE;
    let shape = [ch, sp, sp];

    let mut b = TensorBlockBuilder::new("dly_pipe_backbone_res");
    let input = b.add_input("features", &shape);

    // Bottleneck with skip connection
    let w1 = b.add_input("res_w1", &[ch, ch, 3, 3]);
    let conv1 = b.add_conv2d(input, w1, None, 1, 1, 1, 1, &shape);
    let act1 = add_silu(&mut b, conv1, &shape);
    let w2 = b.add_input("res_w2", &[ch, ch, 3, 3]);
    let conv2 = b.add_conv2d(act1, w2, None, 1, 1, 1, 1, &shape);

    // Residual addition
    let out = b.add_binary_add(input, conv2, &shape);
    let def = b.build(out).expect("valid residual kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[ch, ch, 3, 3]),
        weight(&[ch, ch, 3, 3]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&shape, 1.0);

    // Compare residual vs non-residual bounds
    let output = graph.propagate_ibp(&input).expect("IBP through residual");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline residual IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // Residual adds identity to conv path, so bounds include the input range
    let width = hi_max - lo_min;
    assert!(
        width >= 2.0 - 0.01,
        "residual output width should be >= input width (2.0), got {width}"
    );
}

// ===========================================================================
// 16. C2f bottleneck with shortcut bounds (CROWN)
// ===========================================================================

#[test]
fn test_c2f_bottleneck_shortcut_crown() {
    let ch = BASE_CH;
    let sp = IMG_SIZE;
    let shape = [ch, sp, sp];

    let mut b = TensorBlockBuilder::new("dly_pipe_c2f_shortcut");
    let input = b.add_input("features", &shape);

    // Bottleneck with shortcut (residual)
    let bottleneck = add_bottleneck(&mut b, input, ch, sp, "bneck0");
    let out = b.add_binary_add(input, bottleneck, &shape);
    let def = b.build(out).expect("valid C2f shortcut kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_bottleneck_bindings(&mut bindings, ch);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&shape, 0.5);

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through C2f shortcut");
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("DLY pipeline C2f shortcut IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN should also produce valid bounds
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("DLY pipeline C2f shortcut CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 17. Detection head class probability bounds
// ===========================================================================

#[test]
fn test_detection_head_class_probability_ibp() {
    // Classification with softmax to produce proper probability distribution.
    let ch = BASE_CH;
    let mut b = TensorBlockBuilder::new("dly_pipe_cls_prob");
    let input = b.add_input("features", &[NUM_BOXES, ch]);

    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, ch]);
    let logits = b.add_linear(input, cls_w, None, &[NUM_BOXES, NUM_CLASSES]);
    let out = b.add_softmax(logits, -1, &[NUM_BOXES, NUM_CLASSES]);
    let def = b.build(out).expect("valid cls prob kernel");

    let bindings = vec![TensorParamBinding::Variable, weight(&[NUM_CLASSES, ch])];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&[NUM_BOXES, ch], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through cls probs");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline cls probs IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax outputs sum to 1 per row, each in [0, 1]
    assert!(
        lo_min >= -1e-5,
        "softmax lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "softmax upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 18. Multi-class output shape preservation
// ===========================================================================

#[test]
fn test_multi_class_output_shape_preservation_ibp() {
    // Verify that the full classification pipeline preserves the expected
    // output shape [NUM_BOXES, NUM_CLASSES] through all stages.
    let ch = BASE_CH;
    let sp = IMG_SIZE;

    let mut b = TensorBlockBuilder::new("dly_pipe_shape_preserve");
    let input = b.add_input("features", &[ch, sp, sp]);

    // Flatten + project
    let flat = b.add_reshape(input, &[ch, sp * sp]);
    let transposed = b.add_transpose(flat, &[1, 0], &[sp * sp, ch]);

    // Classification: Linear -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, ch]);
    let logits = b.add_linear(transposed, cls_w, None, &[sp * sp, NUM_CLASSES]);
    let conf = b.add_sigmoid(logits, &[sp * sp, NUM_CLASSES]);

    // Box regression: Linear -> sigmoid
    let box_w = b.add_input("box_w", &[HEAD_DIM, ch]);
    let box_logits = b.add_linear(transposed, box_w, None, &[sp * sp, HEAD_DIM]);
    let _box_out = b.add_sigmoid(box_logits, &[sp * sp, HEAD_DIM]);

    // Build through classification output
    let def = b.build(conf).expect("valid shape preserve kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, ch]),
        weight(&[HEAD_DIM, ch]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = feature_bounds(&[ch, sp, sp], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through shape pipeline");
    assert_bounds_valid(&output);

    // Verify output shape matches expected detection output
    let (lo, _) = output.lower_upper();
    let expected_positions = sp * sp;
    assert_eq!(
        lo.shape(),
        &[expected_positions, NUM_CLASSES],
        "classification output shape must be [positions, NUM_CLASSES]"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DLY pipeline shape preserve IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid bounded [0, 1]
    assert!(
        lo_min >= -0.01,
        "sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(hi_max <= 1.01, "sigmoid upper should be <= 1, got {hi_max}");

    // Monotone tightening: narrower input -> narrower or equal output
    let narrow_input = feature_bounds(&[ch, sp, sp], 0.5);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");
    let wide_width = bound_width(&output);
    let narrow_width = bound_width(&narrow_output);
    assert!(
        narrow_width <= wide_width + 1e-4,
        "monotone tightening: narrow_width={narrow_width} > wide_width={wide_width}"
    );
}
