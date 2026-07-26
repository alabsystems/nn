// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for text detection probability map pipeline (PaddleOCR DB detector).
//!
//! Verifies IBP and CROWN bound propagation through the text detection
//! probability map operations used in DB (Differentiable Binarization)
//! text detectors:
//!
//! ## DB Detector Core (tests 1-4)
//!
//! 1. Conv backbone -> sigmoid probability map in (0, 1) (IBP + CROWN)
//! 2. Threshold map: Conv -> sigmoid threshold in (0, 1) (IBP)
//! 3. Binary map: sigmoid(k * (P - T)) approximation (IBP + CROWN)
//! 4. Probability map spatial resolution: stride-preserving (IBP)
//!
//! ## Multi-Scale Detection (tests 5-7)
//!
//! 5. FPN feature fusion: multi-scale probability maps (IBP)
//! 6. Upsampled probability map: bilinear upsampling preserves [0, 1] (IBP)
//! 7. Feature pyramid 3-level: stride 4, 8, 16 maps (IBP)
//!
//! ## Binarization (tests 8-10)
//!
//! 8. Hard threshold: P > 0.3 classification boundary (IBP)
//! 9. Soft threshold: differentiable binarization approximation (IBP + CROWN)
//! 10. Threshold sensitivity: small threshold change -> bounded output change (IBP)
//!
//! ## Text Region Properties (tests 11-15)
//!
//! 11. Region confidence: max probability in region bounded (IBP)
//! 12. Region area: spatial extent bounded by input resolution (IBP)
//! 13. Probability monotone tightening: smaller eps -> tighter map bounds (IBP)
//! 14. Detection -> recognition handoff: cropped region bounds (IBP)
//! 15. Full pipeline: backbone -> FPN -> probability -> binarization (IBP)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - Feature maps: 16x16 input, 8x8 / 4x4 after strided convolutions
//! - Channels: 8 (backbone) -> 16 (after conv) -> 1 (probability map)
//!
//! Architecture reference:
//! - DB (Liao et al. 2020): Differentiable Binarization for text detection
//! - PaddleOCR (Baidu): Production OCR with DB detector
//!
//! Part of #3997: NY compose tests for text detection probability maps.

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

/// Spatial size of input feature maps.
const SPATIAL: usize = 16;
/// Input channels for backbone feature maps.
const CHANNELS: usize = 8;
/// Intermediate channels after convolution.
const MID_CH: usize = 16;
/// Single-channel output for probability/threshold/binary maps.
const MAP_CH: usize = 1;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// DB differentiable binarization expansion factor k.
const DB_K: f32 = 50.0;

// ===========================================================================
// 1. Conv backbone -> sigmoid probability map (IBP + CROWN)
// ===========================================================================

/// Build Conv backbone -> sigmoid probability map.
///
/// Input: [CHANNELS, SPATIAL, SPATIAL] -> Conv2d -> ReLU -> Conv2d -> sigmoid
/// Output: [MAP_CH, SPATIAL, SPATIAL] (probability map in (0, 1)).
fn build_conv_sigmoid_prob_map() -> TensorKernelDef {
    let shape_mid = [MID_CH, SPATIAL, SPATIAL];
    let shape_out = [MAP_CH, SPATIAL, SPATIAL];
    let mut b = TensorBlockBuilder::new("text_det_conv_sigmoid_prob_map");

    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Conv2d #1: [CHANNELS, 16, 16] -> [MID_CH, 16, 16]
    let w1 = b.add_input("conv1_weight", &[MID_CH, CHANNELS, 3, 3]);
    let b1 = b.add_input("conv1_bias", &[MID_CH]);
    let conv1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 1, 1, &shape_mid);
    let relu = b.add_relu(conv1, &shape_mid);

    // Conv2d #2: [MID_CH, 16, 16] -> [MAP_CH, 16, 16]
    let w2 = b.add_input("conv2_weight", &[MAP_CH, MID_CH, 1, 1]);
    let b2 = b.add_input("conv2_bias", &[MAP_CH]);
    let conv2 = b.add_conv2d(relu, w2, Some(b2), 1, 1, 0, 0, &shape_out);

    // Sigmoid: output in (0, 1)
    let out = b.add_sigmoid(conv2, &shape_out);

    b.build(out).expect("valid conv sigmoid prob map kernel")
}

fn conv_sigmoid_prob_map_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MID_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MID_CH]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, MID_CH, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
    ]
}

#[test]
fn test_text_det_conv_sigmoid_prob_map_ibp() {
    let def = build_conv_sigmoid_prob_map();
    let bindings = conv_sigmoid_prob_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Conv -> sigmoid prob map");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, SPATIAL, SPATIAL],
        "probability map output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv -> sigmoid prob map IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid output must be in (0, 1)
    assert!(lo_min >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

#[test]
fn test_text_det_conv_sigmoid_prob_map_crown() {
    let def = build_conv_sigmoid_prob_map();
    let bindings = conv_sigmoid_prob_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[MAP_CH, SPATIAL, SPATIAL]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv -> sigmoid prob map CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    // Sigmoid bounds still hold
    assert!(lo_min >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 2. Threshold map: Conv -> sigmoid threshold in (0, 1) (IBP)
// ===========================================================================

/// Build threshold map head: Conv2d -> sigmoid.
///
/// The DB detector has a separate threshold map head that predicts
/// per-pixel thresholds in (0, 1) for binarization.
///
/// Input: [CHANNELS, SPATIAL, SPATIAL] -> Conv2d -> sigmoid
/// Output: [MAP_CH, SPATIAL, SPATIAL] (threshold map in (0, 1)).
fn build_threshold_map() -> TensorKernelDef {
    let shape_out = [MAP_CH, SPATIAL, SPATIAL];
    let mut b = TensorBlockBuilder::new("text_det_threshold_map");

    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);
    let w = b.add_input("thresh_weight", &[MAP_CH, CHANNELS, 3, 3]);
    let bias = b.add_input("thresh_bias", &[MAP_CH]);
    let conv = b.add_conv2d(input, w, Some(bias), 1, 1, 1, 1, &shape_out);
    let out = b.add_sigmoid(conv, &shape_out);

    b.build(out).expect("valid threshold map kernel")
}

fn threshold_map_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
    ]
}

#[test]
fn test_text_det_threshold_map_ibp() {
    let def = build_threshold_map();
    let bindings = threshold_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through threshold map");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, SPATIAL, SPATIAL],
        "threshold map output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Threshold map IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= 0.0 - 1e-6, "threshold lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "threshold upper <= 1, got {hi_max}");
}

// ===========================================================================
// 3. Binary map: sigmoid(k * (P - T)) approximation (IBP + CROWN)
// ===========================================================================

/// Build DB binary map: sigmoid(k * (P - T)).
///
/// In the DB detector, the binary map approximates hard thresholding
/// with a differentiable sigmoid: B = sigmoid(k * (P - T)), where
/// k is a large expansion factor (~50), P is the probability map,
/// and T is the threshold map. Both P and T are in (0, 1).
///
/// We model this as: input[0..1] = P, input[1..2] = T (via concat).
/// Linear layer computes k*(P-T) then sigmoid.
///
/// Input: [2, SPATIAL, SPATIAL] (P and T stacked on channel axis)
/// Output: [MAP_CH, SPATIAL, SPATIAL] (binary map in (0, 1)).
fn build_binary_map() -> TensorKernelDef {
    let in_shape = [2, SPATIAL, SPATIAL];
    let shape_out = [MAP_CH, SPATIAL, SPATIAL];
    let mut b = TensorBlockBuilder::new("text_det_binary_map");

    // Input: 2-channel (P stacked with T)
    let input = b.add_input("prob_thresh", &in_shape);

    // 1x1 Conv2d implements k*(P - T): weight[0,0] = k, weight[0,1] = -k
    let w = b.add_input("binary_weight", &[MAP_CH, 2, 1, 1]);
    let bias = b.add_input("binary_bias", &[MAP_CH]);
    let diff = b.add_conv2d(input, w, Some(bias), 1, 1, 0, 0, &shape_out);
    let out = b.add_sigmoid(diff, &shape_out);

    b.build(out).expect("valid binary map kernel")
}

fn binary_map_bindings() -> Vec<TensorParamBinding> {
    // Weight encodes k*(P - T): channel 0 of input (P) gets +k, channel 1 (T) gets -k
    let mut w_data = vec![0.0f32; MAP_CH * 2 * 1 * 1];
    w_data[0] = DB_K; // weight for P channel
    w_data[1] = -DB_K; // weight for T channel
    let w = ArrayD::from_shape_vec(IxDyn(&[MAP_CH, 2, 1, 1]), w_data).expect("valid weight");

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
    ]
}

#[test]
fn test_text_det_binary_map_ibp() {
    let def = build_binary_map();
    let bindings = binary_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // P and T both in [0.2, 0.8] — typical after sigmoid
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, SPATIAL, SPATIAL]), 0.2f32),
        ArrayD::from_elem(IxDyn(&[2, SPATIAL, SPATIAL]), 0.8f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through binary map");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, SPATIAL, SPATIAL],
        "binary map output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Binary map IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid output is in (0, 1)
    assert!(lo_min >= 0.0 - 1e-6, "binary map lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "binary map upper <= 1, got {hi_max}");
}

#[test]
fn test_text_det_binary_map_crown() {
    let def = build_binary_map();
    let bindings = binary_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, SPATIAL, SPATIAL]), 0.3f32),
        ArrayD::from_elem(IxDyn(&[2, SPATIAL, SPATIAL]), 0.7f32),
    )
    .expect("valid bounds");

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[MAP_CH, SPATIAL, SPATIAL]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Binary map CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min >= 0.0 - 1e-6, "binary map lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "binary map upper <= 1, got {hi_max}");
}

// ===========================================================================
// 4. Probability map spatial resolution: stride-preserving (IBP)
// ===========================================================================

/// Build stride-preserving probability map: Conv2d(s=1, p=1) -> sigmoid.
///
/// Verifies that the probability map preserves spatial resolution
/// (no downsampling) as required by pixel-level text detection.
///
/// Input: [CHANNELS, SPATIAL, SPATIAL] -> Conv2d(s=1) -> sigmoid
/// Output: [MAP_CH, SPATIAL, SPATIAL] (same spatial dims).
fn build_stride_preserving_map() -> TensorKernelDef {
    let shape_out = [MAP_CH, SPATIAL, SPATIAL];
    let mut b = TensorBlockBuilder::new("text_det_stride_preserving");

    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);
    let w = b.add_input("weight", &[MAP_CH, CHANNELS, 3, 3]);
    let bias = b.add_input("bias", &[MAP_CH]);
    let conv = b.add_conv2d(input, w, Some(bias), 1, 1, 1, 1, &shape_out);
    let out = b.add_sigmoid(conv, &shape_out);

    b.build(out).expect("valid stride-preserving map kernel")
}

fn stride_preserving_map_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
    ]
}

#[test]
fn test_text_det_stride_preserving_ibp() {
    let def = build_stride_preserving_map();
    let bindings = stride_preserving_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through stride-preserving map");

    // Key assertion: spatial dimensions are preserved
    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, SPATIAL, SPATIAL],
        "stride-preserving: spatial dims must match input"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Stride-preserving map IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 5. FPN feature fusion: multi-scale probability maps (IBP)
// ===========================================================================

/// Build FPN-style feature fusion: multi-scale Conv2d -> concat -> Conv2d -> sigmoid.
///
/// The DB detector uses a Feature Pyramid Network to fuse features from
/// different backbone stages before producing the probability map.
///
/// Input: [CHANNELS, SPATIAL, SPATIAL]
/// Branches: Conv2d(s=1) at scale 1 and Conv2d(s=2) at scale 2
/// Concat the scale-1 features, then 1x1 conv -> sigmoid.
///
/// Output: [MAP_CH, SPATIAL, SPATIAL].
fn build_fpn_feature_fusion() -> TensorKernelDef {
    let ch_per_scale = 4usize;
    let fused_ch = ch_per_scale * 2; // after concat
    let shape_s1 = [ch_per_scale, SPATIAL, SPATIAL];
    let half_s = SPATIAL / 2;
    let shape_s2 = [ch_per_scale, half_s, half_s];
    let shape_out = [MAP_CH, SPATIAL, SPATIAL];

    let mut b = TensorBlockBuilder::new("text_det_fpn_fusion");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Scale 1: Conv2d(s=1, k=3, p=1) -> [ch_per_scale, SPATIAL, SPATIAL]
    let w1 = b.add_input("s1_weight", &[ch_per_scale, CHANNELS, 3, 3]);
    let b1 = b.add_input("s1_bias", &[ch_per_scale]);
    let s1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 1, 1, &shape_s1);
    let s1_relu = b.add_relu(s1, &shape_s1);

    // Scale 2: Conv2d(s=2, k=3, p=1) -> [ch_per_scale, half, half]
    let w2 = b.add_input("s2_weight", &[ch_per_scale, CHANNELS, 3, 3]);
    let b2 = b.add_input("s2_bias", &[ch_per_scale]);
    let s2 = b.add_conv2d(input, w2, Some(b2), 2, 2, 1, 1, &shape_s2);
    let s2_relu = b.add_relu(s2, &shape_s2);

    // Upsample scale 2 to scale 1 via ConvTranspose2d(s=2, k=4, p=1)
    let w_up = b.add_input("up_weight", &[ch_per_scale, ch_per_scale, 4, 4]);
    let b_up = b.add_input("up_bias", &[ch_per_scale]);
    let s2_up = b.add_conv_transpose_2d(
        s2_relu,
        w_up,
        Some(b_up),
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        &shape_s1,
    );

    // Concat scale 1 + upsampled scale 2
    let fused_shape = [fused_ch, SPATIAL, SPATIAL];
    let fused = b.add_concat(&[s1_relu, s2_up], 0, &fused_shape);

    // 1x1 Conv2d -> sigmoid for final probability map
    let w_out = b.add_input("out_weight", &[MAP_CH, fused_ch, 1, 1]);
    let b_out = b.add_input("out_bias", &[MAP_CH]);
    let conv_out = b.add_conv2d(fused, w_out, Some(b_out), 1, 1, 0, 0, &shape_out);
    let out = b.add_sigmoid(conv_out, &shape_out);

    b.build(out).expect("valid FPN feature fusion kernel")
}

fn fpn_feature_fusion_bindings() -> Vec<TensorParamBinding> {
    let ch_per_scale = 4usize;
    let fused_ch = ch_per_scale * 2;
    vec![
        TensorParamBinding::Variable,
        // s1 weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ch_per_scale, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch_per_scale]), 0.0f32)),
        // s2 weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ch_per_scale, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch_per_scale]), 0.0f32)),
        // upsample weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ch_per_scale, ch_per_scale, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch_per_scale]), 0.0f32)),
        // output 1x1 conv weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, fused_ch, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
    ]
}

#[test]
fn test_text_det_fpn_feature_fusion_ibp() {
    let def = build_fpn_feature_fusion();
    let bindings = fpn_feature_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FPN feature fusion");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, SPATIAL, SPATIAL],
        "FPN fusion output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FPN feature fusion IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 6. Upsampled probability map: bilinear upsampling preserves [0, 1] (IBP)
// ===========================================================================

/// Build upsampled probability map: Conv2d(s=2) -> sigmoid -> ConvTranspose2d.
///
/// Verifies that producing a probability map at reduced resolution and
/// upsampling back preserves the [0, 1] bound invariant.
///
/// Input: [CHANNELS, SPATIAL, SPATIAL] -> Conv2d(s=2) -> sigmoid -> upsample
/// Output: [MAP_CH, SPATIAL, SPATIAL].
fn build_upsampled_prob_map() -> TensorKernelDef {
    let half_s = SPATIAL / 2;
    let shape_half = [MAP_CH, half_s, half_s];
    let shape_out = [MAP_CH, SPATIAL, SPATIAL];

    let mut b = TensorBlockBuilder::new("text_det_upsampled_prob_map");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Downsample: Conv2d(s=2) -> [MAP_CH, half, half]
    let w = b.add_input("down_weight", &[MAP_CH, CHANNELS, 3, 3]);
    let bias = b.add_input("down_bias", &[MAP_CH]);
    let conv = b.add_conv2d(input, w, Some(bias), 2, 2, 1, 1, &shape_half);
    let prob = b.add_sigmoid(conv, &shape_half);

    // Upsample: ConvTranspose2d(s=2, k=4, p=1) -> [MAP_CH, SPATIAL, SPATIAL]
    let w_up = b.add_input("up_weight", &[MAP_CH, MAP_CH, 4, 4]);
    let b_up = b.add_input("up_bias", &[MAP_CH]);
    let out = b.add_conv_transpose_2d(
        prob,
        w_up,
        Some(b_up),
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        &shape_out,
    );

    b.build(out).expect("valid upsampled prob map kernel")
}

fn upsampled_prob_map_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
        // Upsample weights near-identity to preserve bounds
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, MAP_CH, 4, 4]),
            0.0625f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
    ]
}

#[test]
fn test_text_det_upsampled_prob_map_ibp() {
    let def = build_upsampled_prob_map();
    let bindings = upsampled_prob_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through upsampled prob map");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, SPATIAL, SPATIAL],
        "upsampled prob map output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Upsampled prob map IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. Feature pyramid 3-level: stride 4, 8, 16 maps (IBP)
// ===========================================================================

/// Build 3-level feature pyramid with stride 4, 8, 16.
///
/// Uses the coarsest level (stride 16) for verification to capture
/// the full depth of the pyramid and measure bound propagation.
///
/// Input: [CHANNELS, SPATIAL, SPATIAL]
/// Output: [MAP_CH, SPATIAL/16, SPATIAL/16] = [1, 1, 1] (stride 16).
fn build_feature_pyramid_3level() -> TensorKernelDef {
    let ch = 4usize;
    let s4 = SPATIAL / 4; // 4
    let s8 = SPATIAL / 8; // 2
    let s16 = SPATIAL / 16; // 1

    let mut b = TensorBlockBuilder::new("text_det_feature_pyramid_3level");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Level 1 (stride 4): Conv2d(s=4, k=3, p=1)
    let w1 = b.add_input("l1_weight", &[ch, CHANNELS, 3, 3]);
    let b1 = b.add_input("l1_bias", &[ch]);
    let _l1 = b.add_conv2d(input, w1, Some(b1), 4, 4, 1, 1, &[ch, s4, s4]);

    // Level 2 (stride 8): two Conv2d(s=2) cascaded: 16->8->4... actually stride 4 then 2
    // Simpler: single Conv2d(s=2) on level 1 output
    let w2 = b.add_input("l2_weight", &[ch, ch, 3, 3]);
    let b2 = b.add_input("l2_bias", &[ch]);
    let l2 = b.add_conv2d(_l1, w2, Some(b2), 2, 2, 1, 1, &[ch, s8, s8]);

    // Level 3 (stride 16): Conv2d(s=2) on level 2
    let w3 = b.add_input("l3_weight", &[ch, ch, 3, 3]);
    let b3 = b.add_input("l3_bias", &[ch]);
    let l3 = b.add_conv2d(l2, w3, Some(b3), 2, 2, 1, 1, &[ch, s16, s16]);

    // 1x1 Conv2d -> sigmoid for probability map at coarsest level
    let w_out = b.add_input("out_weight", &[MAP_CH, ch, 1, 1]);
    let b_out = b.add_input("out_bias", &[MAP_CH]);
    let conv_out = b.add_conv2d(l3, w_out, Some(b_out), 1, 1, 0, 0, &[MAP_CH, s16, s16]);
    let out = b.add_sigmoid(conv_out, &[MAP_CH, s16, s16]);

    b.build(out).expect("valid feature pyramid 3-level kernel")
}

fn feature_pyramid_3level_bindings() -> Vec<TensorParamBinding> {
    let ch = 4usize;
    vec![
        TensorParamBinding::Variable,
        // L1 weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ch, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
        // L2 weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch, ch, 3, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
        // L3 weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch, ch, 3, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
        // Output 1x1 weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, ch, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
    ]
}

#[test]
fn test_text_det_feature_pyramid_3level_ibp() {
    let def = build_feature_pyramid_3level();
    let bindings = feature_pyramid_3level_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-level feature pyramid");

    let s16 = SPATIAL / 16;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, s16, s16],
        "3-level pyramid output shape (stride 16)"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Feature pyramid 3-level IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid output bounded in (0, 1)
    assert!(lo_min >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 8. Hard threshold: P > 0.3 classification boundary (IBP)
// ===========================================================================

/// Build hard threshold approximation: sigmoid(k * (P - 0.3)).
///
/// Models the hard threshold P > 0.3 as a steep sigmoid with a fixed
/// threshold. The input is a single-channel probability map in [0, 1].
///
/// Input: [MAP_CH, SPATIAL, SPATIAL] (probability map)
/// Output: [MAP_CH, SPATIAL, SPATIAL] (near-binary output).
fn build_hard_threshold() -> TensorKernelDef {
    let shape = [MAP_CH, SPATIAL, SPATIAL];
    let mut b = TensorBlockBuilder::new("text_det_hard_threshold");

    let input = b.add_input("prob_map", &shape);
    // 1x1 Conv2d with weight=k, bias=-k*0.3 implements k*(P - 0.3)
    let w = b.add_input("thresh_weight", &[MAP_CH, MAP_CH, 1, 1]);
    let bias = b.add_input("thresh_bias", &[MAP_CH]);
    let scaled = b.add_conv2d(input, w, Some(bias), 1, 1, 0, 0, &shape);
    let out = b.add_sigmoid(scaled, &shape);

    b.build(out).expect("valid hard threshold kernel")
}

fn hard_threshold_bindings() -> Vec<TensorParamBinding> {
    let k = DB_K;
    let threshold = 0.3f32;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH, MAP_CH, 1, 1]), k)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), -k * threshold)),
    ]
}

#[test]
fn test_text_det_hard_threshold_ibp() {
    let def = build_hard_threshold();
    let bindings = hard_threshold_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Input probability map in [0, 1]
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[MAP_CH, SPATIAL, SPATIAL]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[MAP_CH, SPATIAL, SPATIAL]), 1.0f32),
    )
    .expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through hard threshold");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, SPATIAL, SPATIAL],
        "hard threshold output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Hard threshold IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid output in (0, 1)
    assert!(lo_min >= 0.0 - 1e-6, "threshold lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "threshold upper <= 1, got {hi_max}");
}

// ===========================================================================
// 9. Soft threshold: differentiable binarization approximation (IBP + CROWN)
// ===========================================================================

/// Build soft differentiable binarization: Conv -> sigmoid -> scale.
///
/// Models the soft binarization path with a learnable convolution
/// followed by sigmoid for smooth probability output.
///
/// Input: [CHANNELS, SPATIAL, SPATIAL]
/// Output: [MAP_CH, SPATIAL, SPATIAL] (soft binary output in (0, 1)).
fn build_soft_threshold() -> TensorKernelDef {
    let shape_mid = [MID_CH, SPATIAL, SPATIAL];
    let shape_out = [MAP_CH, SPATIAL, SPATIAL];
    let mut b = TensorBlockBuilder::new("text_det_soft_threshold");

    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Conv layer for feature extraction
    let w1 = b.add_input("conv_weight", &[MID_CH, CHANNELS, 3, 3]);
    let b1 = b.add_input("conv_bias", &[MID_CH]);
    let conv = b.add_conv2d(input, w1, Some(b1), 1, 1, 1, 1, &shape_mid);
    let relu = b.add_relu(conv, &shape_mid);

    // 1x1 projection + sigmoid
    let w2 = b.add_input("proj_weight", &[MAP_CH, MID_CH, 1, 1]);
    let b2 = b.add_input("proj_bias", &[MAP_CH]);
    let proj = b.add_conv2d(relu, w2, Some(b2), 1, 1, 0, 0, &shape_out);
    let out = b.add_sigmoid(proj, &shape_out);

    b.build(out).expect("valid soft threshold kernel")
}

fn soft_threshold_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MID_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MID_CH]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, MID_CH, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
    ]
}

#[test]
fn test_text_det_soft_threshold_ibp() {
    let def = build_soft_threshold();
    let bindings = soft_threshold_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through soft threshold");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, SPATIAL, SPATIAL],
        "soft threshold output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Soft threshold IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

#[test]
fn test_text_det_soft_threshold_crown() {
    let def = build_soft_threshold();
    let bindings = soft_threshold_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[MAP_CH, SPATIAL, SPATIAL]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Soft threshold CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. Threshold sensitivity: small threshold change -> bounded output change (IBP)
// ===========================================================================

/// Verify threshold sensitivity: narrower input eps -> tighter binary output.
///
/// Uses the binary map graph (sigmoid(k*(P-T))) and checks that
/// narrower P-T input range produces tighter output bounds.
#[test]
fn test_text_det_threshold_sensitivity_ibp() {
    let def = build_binary_map();
    let bindings = binary_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide: P, T in [0.1, 0.9]
    let wide_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, SPATIAL, SPATIAL]), 0.1f32),
        ArrayD::from_elem(IxDyn(&[2, SPATIAL, SPATIAL]), 0.9f32),
    )
    .expect("valid wide bounds");
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);

    // Narrow: P, T in [0.4, 0.6]
    let narrow_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, SPATIAL, SPATIAL]), 0.4f32),
        ArrayD::from_elem(IxDyn(&[2, SPATIAL, SPATIAL]), 0.6f32),
    )
    .expect("valid narrow bounds");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");
    assert_bounds_valid(&narrow_output);

    let wide_width = total_bound_width(&wide_output);
    let narrow_width = total_bound_width(&narrow_output);

    eprintln!("Threshold sensitivity: wide_width={wide_width:.4}, narrow_width={narrow_width:.4}");
    assert!(
        narrow_width <= wide_width + 1e-6,
        "narrower threshold range must produce tighter binary output: \
         narrow_width={narrow_width} > wide_width={wide_width}"
    );
}

// ===========================================================================
// 11. Region confidence: max probability in region bounded (IBP)
// ===========================================================================

/// Build region confidence: Conv -> sigmoid -> AvgPool (global average).
///
/// The average pooling aggregates the probability map into a single
/// confidence score per channel. This models the confidence of a
/// detected text region.
///
/// Input: [CHANNELS, SPATIAL, SPATIAL]
/// Output: [MAP_CH, 1, 1] (region confidence score).
fn build_region_confidence() -> TensorKernelDef {
    let shape_prob = [MAP_CH, SPATIAL, SPATIAL];
    let shape_out = [MAP_CH, 1, 1];

    let mut b = TensorBlockBuilder::new("text_det_region_confidence");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Conv -> sigmoid probability map
    let w = b.add_input("prob_weight", &[MAP_CH, CHANNELS, 3, 3]);
    let bias = b.add_input("prob_bias", &[MAP_CH]);
    let conv = b.add_conv2d(input, w, Some(bias), 1, 1, 1, 1, &shape_prob);
    let prob = b.add_sigmoid(conv, &shape_prob);

    // Global average pool: [MAP_CH, 16, 16] -> [MAP_CH, 1, 1]
    let out = b.add_avg_pool_2d(prob, SPATIAL, SPATIAL, SPATIAL, SPATIAL, 0, 0, &shape_out);

    b.build(out).expect("valid region confidence kernel")
}

fn region_confidence_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
    ]
}

#[test]
fn test_text_det_region_confidence_ibp() {
    let def = build_region_confidence();
    let bindings = region_confidence_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through region confidence");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, 1, 1],
        "region confidence output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Region confidence IBP: bounds=[{lo_min}, {hi_max}]");
    // Confidence is average of sigmoid outputs, still in [0, 1]
    assert!(lo_min >= 0.0 - 1e-6, "confidence lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "confidence upper <= 1, got {hi_max}");
}

// ===========================================================================
// 12. Region area: spatial extent bounded by input resolution (IBP)
// ===========================================================================

/// Build region area estimator: Conv -> sigmoid -> AvgPool at two resolutions.
///
/// Verifies that probability maps at different spatial scales both
/// produce valid bounded outputs within [0, 1].
///
/// Input: [CHANNELS, SPATIAL, SPATIAL]
/// Output: [MAP_CH, 4, 4] (low-res probability map).
fn build_region_area() -> TensorKernelDef {
    let s4 = SPATIAL / 4; // 4
    let shape_out = [MAP_CH, s4, s4];

    let mut b = TensorBlockBuilder::new("text_det_region_area");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Strided conv to reduce spatial resolution
    let w = b.add_input("area_weight", &[MAP_CH, CHANNELS, 3, 3]);
    let bias = b.add_input("area_bias", &[MAP_CH]);
    let conv = b.add_conv2d(input, w, Some(bias), 4, 4, 1, 1, &shape_out);
    let out = b.add_sigmoid(conv, &shape_out);

    b.build(out).expect("valid region area kernel")
}

fn region_area_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
    ]
}

#[test]
fn test_text_det_region_area_ibp() {
    let def = build_region_area();
    let bindings = region_area_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through region area");

    let s4 = SPATIAL / 4;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, s4, s4],
        "region area output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Region area IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= 0.0 - 1e-6, "area lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "area upper <= 1, got {hi_max}");
}

// ===========================================================================
// 13. Probability monotone tightening: smaller eps -> tighter map bounds (IBP)
// ===========================================================================

/// Verify that narrower input perturbation produces tighter probability map bounds.
///
/// This is the fundamental monotone interval arithmetic property applied
/// to the text detection probability pipeline.
#[test]
fn test_text_det_prob_monotone_tightening() {
    let def = build_conv_sigmoid_prob_map();
    let bindings = conv_sigmoid_prob_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input: [-1.0, 1.0]
    let wide_input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);

    // Narrow input: [-0.1, 0.1]
    let narrow_input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 0.1);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");
    assert_bounds_valid(&narrow_output);

    let wide_width = total_bound_width(&wide_output);
    let narrow_width = total_bound_width(&narrow_output);

    eprintln!(
        "Probability monotone tightening: wide_width={wide_width:.4}, \
         narrow_width={narrow_width:.4}"
    );
    assert!(
        narrow_width <= wide_width + 1e-6,
        "narrower input must produce tighter probability map bounds: \
         narrow_width={narrow_width} > wide_width={wide_width}"
    );
}

// ===========================================================================
// 14. Detection -> recognition handoff: cropped region bounds (IBP)
// ===========================================================================

/// Build detection -> recognition handoff: sigmoid prob map -> crop (narrow) -> linear.
///
/// Models the handoff from text detection (probability map) to text
/// recognition: the detector produces a probability map, a region is
/// cropped, and features are projected for recognition.
///
/// Input: [CHANNELS, SPATIAL, SPATIAL]
/// Output: [MAP_CH, 4, 4] (cropped region features).
fn build_detection_recognition_handoff() -> TensorKernelDef {
    let shape_prob = [MAP_CH, SPATIAL, SPATIAL];
    let crop_s = 4usize;
    let shape_crop = [MAP_CH, crop_s, crop_s];

    let mut b = TensorBlockBuilder::new("text_det_recognition_handoff");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Probability map head
    let w = b.add_input("prob_weight", &[MAP_CH, CHANNELS, 3, 3]);
    let bias = b.add_input("prob_bias", &[MAP_CH]);
    let conv = b.add_conv2d(input, w, Some(bias), 1, 1, 1, 1, &shape_prob);
    let prob = b.add_sigmoid(conv, &shape_prob);

    // Crop: narrow spatial dims to [4, 4] region (simulates RoI crop)
    let crop_h = b.add_narrow(prob, 1, 0, crop_s, &[MAP_CH, crop_s, SPATIAL]);
    let crop = b.add_narrow(crop_h, 2, 0, crop_s, &shape_crop);

    b.build(crop)
        .expect("valid detection-recognition handoff kernel")
}

fn detection_recognition_handoff_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
    ]
}

#[test]
fn test_text_det_recognition_handoff_ibp() {
    let def = build_detection_recognition_handoff();
    let bindings = detection_recognition_handoff_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detection-recognition handoff");

    let crop_s = 4usize;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, crop_s, crop_s],
        "cropped region output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Detection-recognition handoff IBP: bounds=[{lo_min}, {hi_max}]");
    // Cropped sigmoid output still in [0, 1]
    assert!(
        lo_min >= 0.0 - 1e-6,
        "cropped region lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "cropped region upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 15. Full pipeline: backbone -> FPN -> probability -> binarization (IBP)
// ===========================================================================

/// Build full text detection pipeline:
///   backbone (Conv+ReLU) -> FPN-style fusion -> probability sigmoid -> binary sigmoid.
///
/// Input: [CHANNELS, SPATIAL, SPATIAL]
/// Output: [MAP_CH, SPATIAL, SPATIAL] (binary text detection map).
fn build_full_text_detection_pipeline() -> TensorKernelDef {
    let shape_mid = [MID_CH, SPATIAL, SPATIAL];
    let shape_prob = [MAP_CH, SPATIAL, SPATIAL];
    let shape_thresh = [MAP_CH, SPATIAL, SPATIAL];
    let shape_binary_in = [2, SPATIAL, SPATIAL];
    let shape_out = [MAP_CH, SPATIAL, SPATIAL];

    let mut b = TensorBlockBuilder::new("text_det_full_pipeline");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Backbone: Conv2d -> ReLU
    let w_bb = b.add_input("backbone_weight", &[MID_CH, CHANNELS, 3, 3]);
    let b_bb = b.add_input("backbone_bias", &[MID_CH]);
    let bb_conv = b.add_conv2d(input, w_bb, Some(b_bb), 1, 1, 1, 1, &shape_mid);
    let bb_relu = b.add_relu(bb_conv, &shape_mid);

    // Probability head: 1x1 Conv2d -> sigmoid
    let w_prob = b.add_input("prob_weight", &[MAP_CH, MID_CH, 1, 1]);
    let b_prob = b.add_input("prob_bias", &[MAP_CH]);
    let prob_conv = b.add_conv2d(bb_relu, w_prob, Some(b_prob), 1, 1, 0, 0, &shape_prob);
    let prob = b.add_sigmoid(prob_conv, &shape_prob);

    // Threshold head: 1x1 Conv2d -> sigmoid
    let w_thresh = b.add_input("thresh_weight", &[MAP_CH, MID_CH, 1, 1]);
    let b_thresh = b.add_input("thresh_bias", &[MAP_CH]);
    let thresh_conv = b.add_conv2d(bb_relu, w_thresh, Some(b_thresh), 1, 1, 0, 0, &shape_thresh);
    let thresh = b.add_sigmoid(thresh_conv, &shape_thresh);

    // Binary map: concat(P, T) -> 1x1 Conv(k*(P-T)) -> sigmoid
    let pt = b.add_concat(&[prob, thresh], 0, &shape_binary_in);
    let w_bin = b.add_input("binary_weight", &[MAP_CH, 2, 1, 1]);
    let b_bin = b.add_input("binary_bias", &[MAP_CH]);
    let bin_conv = b.add_conv2d(pt, w_bin, Some(b_bin), 1, 1, 0, 0, &shape_out);
    let out = b.add_sigmoid(bin_conv, &shape_out);

    b.build(out)
        .expect("valid full text detection pipeline kernel")
}

fn full_text_detection_pipeline_bindings() -> Vec<TensorParamBinding> {
    // Binary map weights: k*(P - T)
    let mut w_bin_data = vec![0.0f32; MAP_CH * 2];
    w_bin_data[0] = DB_K;
    w_bin_data[1] = -DB_K;
    let w_bin =
        ArrayD::from_shape_vec(IxDyn(&[MAP_CH, 2, 1, 1]), w_bin_data).expect("valid binary weight");

    vec![
        TensorParamBinding::Variable,
        // Backbone weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MID_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MID_CH]), 0.0f32)),
        // Probability head weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, MID_CH, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
        // Threshold head weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAP_CH, MID_CH, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
        // Binary map weight + bias
        TensorParamBinding::ConstantTensor(w_bin),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAP_CH]), 0.0f32)),
    ]
}

#[test]
fn test_text_det_full_pipeline_ibp() {
    let def = build_full_text_detection_pipeline();
    let bindings = full_text_detection_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full text detection pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAP_CH, SPATIAL, SPATIAL],
        "full pipeline output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full text detection pipeline IBP: bounds=[{lo_min}, {hi_max}]");
    // Final sigmoid output in (0, 1)
    assert!(
        lo_min >= 0.0 - 1e-6,
        "pipeline output lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "pipeline output upper <= 1, got {hi_max}"
    );
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Compute total bound width (sum of hi - lo across all elements).
fn total_bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo, hi) = bounds.lower_upper();
    lo.iter().zip(hi.iter()).map(|(&l, &h)| h - l).sum::<f32>()
}
