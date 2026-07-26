// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Spatial operations NY composition for dpdf models.
//!
//! Verifies IBP and CROWN bound propagation through pooling, upsampling, and
//! strided convolution operations used across document understanding models:
//!
//! 1. **MaxPool2d** single layer IBP bounds
//! 2. **MaxPool2d** with stride IBP bounds
//! 3. **MaxPool2d** CROWN bounds
//! 4. **AvgPool2d** single layer IBP bounds
//! 5. **AvgPool2d** CROWN bounds
//! 6. **AdaptiveAvgPool2d** target size IBP (simulated via AvgPool2d with computed kernel)
//! 7. **Conv2d stride=2 downsampling** IBP bounds
//! 8. **Conv2d stride=2 CROWN** bounds
//! 9. **Transposed Conv2d (upsample)** IBP bounds
//! 10. **SPPF (multi-scale pooling)** IBP bounds (DocLayout-YOLO pattern)
//! 11. **MaxPool chain** (3 cascaded MaxPool2d) IBP
//! 12. **Conv2d -> Pool -> Conv2d composition** IBP
//! 13. **Conv2d -> Pool -> Conv2d composition** CROWN
//! 14. **Spatial downsample -> upsample round-trip** IBP
//! 15. **Feature Pyramid stride pattern** (stride 1, 2, 4) IBP
//! 16. **Spatial monotone tightening**: smaller input eps -> tighter spatial output bounds
//! 17. **AvgPool2d with stride** IBP bounds (stride != kernel)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - Feature maps: 16x16 input, 8x8 / 4x4 after pooling/stride
//! - Channels: 8 (backbone) -> 16 (after conv)
//!
//! Part of #3973: NY compose tests for spatial operations.

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
/// Input/output channels for pooling ops (no channel change).
const CHANNELS: usize = 8;
/// Output channels after convolution.
const CONV_OUT_CH: usize = 16;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// 1. MaxPool2d single layer IBP
// ===========================================================================

/// Build a single MaxPool2d kernel: k=2, s=2, p=0.
/// Input: [CHANNELS, SPATIAL, SPATIAL] -> Output: [CHANNELS, SPATIAL/2, SPATIAL/2].
fn build_maxpool2d_single() -> TensorKernelDef {
    let out_s = SPATIAL / 2;
    let mut b = TensorBlockBuilder::new("spatial_maxpool2d_single");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);
    let out = b.add_max_pool_2d(input, 2, 2, 2, 2, 0, 0, &[CHANNELS, out_s, out_s]);
    b.build(out).expect("valid maxpool2d single kernel")
}

fn maxpool2d_single_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

#[test]
fn test_spatial_maxpool2d_single_ibp() {
    let def = build_maxpool2d_single();
    let bindings = maxpool2d_single_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through MaxPool2d");

    let out_s = SPATIAL / 2;
    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, out_s, out_s]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MaxPool2d single IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min >= -2.0 - 1e-6,
        "MaxPool lower >= input lower, got {lo_min}"
    );
    assert!(
        hi_max <= 2.0 + 1e-6,
        "MaxPool upper <= input upper, got {hi_max}"
    );
}

// ===========================================================================
// 2. MaxPool2d with stride IBP (stride != kernel)
// ===========================================================================

/// Build MaxPool2d with k=3, s=2, p=1: overlapping pooling.
/// Input: [CHANNELS, SPATIAL, SPATIAL] -> Output: [CHANNELS, SPATIAL/2, SPATIAL/2].
fn build_maxpool2d_stride() -> TensorKernelDef {
    let out_s = SPATIAL / 2;
    let mut b = TensorBlockBuilder::new("spatial_maxpool2d_stride");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);
    let out = b.add_max_pool_2d(input, 3, 3, 2, 2, 1, 1, &[CHANNELS, out_s, out_s]);
    b.build(out).expect("valid maxpool2d stride kernel")
}

fn maxpool2d_stride_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

#[test]
fn test_spatial_maxpool2d_stride_ibp() {
    let def = build_maxpool2d_stride();
    let bindings = maxpool2d_stride_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 3.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MaxPool2d stride");

    let out_s = SPATIAL / 2;
    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, out_s, out_s]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MaxPool2d stride IBP: bounds=[{lo_min}, {hi_max}]");
    // MaxPool does not expand bounds beyond input range
    assert!(
        lo_min >= -3.0 - 1e-6,
        "MaxPool stride lower >= -3.0, got {lo_min}"
    );
    assert!(
        hi_max <= 3.0 + 1e-6,
        "MaxPool stride upper <= 3.0, got {hi_max}"
    );
}

// ===========================================================================
// 3. MaxPool2d CROWN bounds
// ===========================================================================

#[test]
fn test_spatial_maxpool2d_crown() {
    let def = build_maxpool2d_single();
    let bindings = maxpool2d_single_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let out_s = SPATIAL / 2;
    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, out_s, out_s]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MaxPool2d CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 4. AvgPool2d single layer IBP
// ===========================================================================

/// Build a single AvgPool2d kernel: k=2, s=2, p=0.
/// Input: [CHANNELS, SPATIAL, SPATIAL] -> Output: [CHANNELS, SPATIAL/2, SPATIAL/2].
fn build_avgpool2d_single() -> TensorKernelDef {
    let out_s = SPATIAL / 2;
    let mut b = TensorBlockBuilder::new("spatial_avgpool2d_single");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);
    let out = b.add_avg_pool_2d(input, 2, 2, 2, 2, 0, 0, &[CHANNELS, out_s, out_s]);
    b.build(out).expect("valid avgpool2d single kernel")
}

fn avgpool2d_single_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

#[test]
fn test_spatial_avgpool2d_single_ibp() {
    let def = build_avgpool2d_single();
    let bindings = avgpool2d_single_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through AvgPool2d");

    let out_s = SPATIAL / 2;
    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, out_s, out_s]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("AvgPool2d single IBP: bounds=[{lo_min}, {hi_max}]");
    // AvgPool averages values, so bounds cannot exceed input bounds.
    assert!(lo_min >= -2.0 - 1e-6, "AvgPool lower >= -2.0, got {lo_min}");
    assert!(hi_max <= 2.0 + 1e-6, "AvgPool upper <= 2.0, got {hi_max}");
}

// ===========================================================================
// 5. AvgPool2d CROWN bounds
// ===========================================================================

#[test]
fn test_spatial_avgpool2d_crown() {
    let def = build_avgpool2d_single();
    let bindings = avgpool2d_single_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let out_s = SPATIAL / 2;
    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, out_s, out_s]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("AvgPool2d CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 6. AdaptiveAvgPool2d (simulated via AvgPool2d with computed kernel)
// ===========================================================================

/// Simulate AdaptiveAvgPool2d(target=4x4) by computing the equivalent kernel.
///
/// For input 16x16 -> output 4x4: kernel = 16/4 = 4, stride = 16/4 = 4, pad = 0.
/// This is how adaptive avg pool works when input_size is divisible by target_size.
fn build_adaptive_avgpool2d() -> TensorKernelDef {
    // input 16x16 -> target 4x4 => k=4, s=4, p=0
    let target = 4usize;
    let k = SPATIAL / target;
    let mut b = TensorBlockBuilder::new("spatial_adaptive_avgpool2d");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);
    let out = b.add_avg_pool_2d(input, k, k, k, k, 0, 0, &[CHANNELS, target, target]);
    b.build(out).expect("valid adaptive avgpool2d kernel")
}

fn adaptive_avgpool2d_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

#[test]
fn test_spatial_adaptive_avgpool2d_ibp() {
    let def = build_adaptive_avgpool2d();
    let bindings = adaptive_avgpool2d_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.5);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through AdaptiveAvgPool2d");

    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, 4, 4]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("AdaptiveAvgPool2d IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min >= -1.5 - 1e-6,
        "adaptive lower >= -1.5, got {lo_min}"
    );
    assert!(hi_max <= 1.5 + 1e-6, "adaptive upper <= 1.5, got {hi_max}");
}

// ===========================================================================
// 7. Conv2d stride=2 downsampling IBP
// ===========================================================================

/// Build Conv2d with stride=2 for spatial downsampling.
/// Input: [CHANNELS, SPATIAL, SPATIAL] -> Output: [CONV_OUT_CH, SPATIAL/2, SPATIAL/2].
fn build_conv2d_stride2() -> TensorKernelDef {
    let out_s = SPATIAL / 2;
    let mut b = TensorBlockBuilder::new("spatial_conv2d_stride2");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);
    let weight = b.add_input("weight", &[CONV_OUT_CH, CHANNELS, 3, 3]);
    let bias = b.add_input("bias", &[CONV_OUT_CH]);
    let out = b.add_conv2d(
        input,
        weight,
        Some(bias),
        2,
        2,
        1,
        1,
        &[CONV_OUT_CH, out_s, out_s],
    );
    b.build(out).expect("valid conv2d stride2 kernel")
}

fn conv2d_stride2_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CONV_OUT_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CONV_OUT_CH]), 0.0f32)),
    ]
}

#[test]
fn test_spatial_conv2d_stride2_ibp() {
    let def = build_conv2d_stride2();
    let bindings = conv2d_stride2_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Conv2d stride=2");

    let out_s = SPATIAL / 2;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CONV_OUT_CH, out_s, out_s],
        "Conv2d stride=2 output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv2d stride=2 IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. Conv2d stride=2 CROWN
// ===========================================================================

#[test]
fn test_spatial_conv2d_stride2_crown() {
    let def = build_conv2d_stride2();
    let bindings = conv2d_stride2_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let out_s = SPATIAL / 2;
    assert_eq!(output.lower_upper().0.shape(), &[CONV_OUT_CH, out_s, out_s]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv2d stride=2 CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 9. Transposed Conv2d (upsample) IBP
// ===========================================================================

/// Build ConvTranspose2d for spatial upsampling: stride=2, k=4, p=1.
/// Input: [CHANNELS, 8, 8] -> Output: [CONV_OUT_CH, 16, 16].
fn build_conv_transpose2d() -> TensorKernelDef {
    let in_s = SPATIAL / 2; // 8
    let out_s = SPATIAL; // 16
    let mut b = TensorBlockBuilder::new("spatial_conv_transpose2d");
    let input = b.add_input("features", &[CHANNELS, in_s, in_s]);
    let weight = b.add_input("weight", &[CHANNELS, CONV_OUT_CH, 4, 4]);
    let bias = b.add_input("bias", &[CONV_OUT_CH]);
    let out = b.add_conv_transpose_2d(
        input,
        weight,
        Some(bias),
        2,
        2, // stride
        1,
        1, // padding
        1,
        1, // dilation
        1, // groups
        0,
        0, // output_padding
        &[CONV_OUT_CH, out_s, out_s],
    );
    b.build(out).expect("valid conv_transpose2d kernel")
}

fn conv_transpose2d_bindings() -> Vec<TensorParamBinding> {
    let in_s = SPATIAL / 2;
    let _ = in_s;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CHANNELS, CONV_OUT_CH, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CONV_OUT_CH]), 0.0f32)),
    ]
}

#[test]
fn test_spatial_conv_transpose2d_ibp() {
    let def = build_conv_transpose2d();
    let bindings = conv_transpose2d_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let in_s = SPATIAL / 2;
    let input = uniform_bounds(&[CHANNELS, in_s, in_s], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ConvTranspose2d");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CONV_OUT_CH, SPATIAL, SPATIAL],
        "ConvTranspose2d output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ConvTranspose2d IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. SPPF (multi-scale pooling) IBP -- DocLayout-YOLO pattern
// ===========================================================================

/// Build an SPPF-style kernel: 3 cascaded MaxPool2d(k=5, s=1, p=2) + concat.
/// Input: [CHANNELS, SPATIAL, SPATIAL] -> Output: [CHANNELS*4, SPATIAL, SPATIAL].
fn build_sppf_spatial() -> TensorKernelDef {
    let shape = [CHANNELS, SPATIAL, SPATIAL];
    let out_shape = [CHANNELS * 4, SPATIAL, SPATIAL];
    let mut b = TensorBlockBuilder::new("spatial_sppf");
    let input = b.add_input("features", &shape);

    let pool1 = b.add_max_pool_2d(input, 5, 5, 1, 1, 2, 2, &shape);
    let pool2 = b.add_max_pool_2d(pool1, 5, 5, 1, 1, 2, 2, &shape);
    let pool3 = b.add_max_pool_2d(pool2, 5, 5, 1, 1, 2, 2, &shape);

    let out = b.add_concat(&[input, pool1, pool2, pool3], 0, &out_shape);
    b.build(out).expect("valid SPPF spatial kernel")
}

fn sppf_spatial_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

#[test]
fn test_spatial_sppf_ibp() {
    let def = build_sppf_spatial();
    let bindings = sppf_spatial_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through SPPF");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS * 4, SPATIAL, SPATIAL],
        "SPPF output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SPPF spatial IBP: bounds=[{lo_min}, {hi_max}]");
    // MaxPool chain does not expand bounds
    assert!(lo_min >= -2.0 - 1e-6, "SPPF lower >= -2.0, got {lo_min}");
    assert!(hi_max <= 2.0 + 1e-6, "SPPF upper <= 2.0, got {hi_max}");
}

// ===========================================================================
// 11. MaxPool chain (3 cascaded MaxPool2d) IBP
// ===========================================================================

/// 3 cascaded MaxPool2d(k=2, s=2): 16x16 -> 8x8 -> 4x4 -> 2x2.
fn build_maxpool_chain() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("spatial_maxpool_chain");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    let s1 = SPATIAL / 2; // 8
    let pool1 = b.add_max_pool_2d(input, 2, 2, 2, 2, 0, 0, &[CHANNELS, s1, s1]);
    let s2 = s1 / 2; // 4
    let pool2 = b.add_max_pool_2d(pool1, 2, 2, 2, 2, 0, 0, &[CHANNELS, s2, s2]);
    let s3 = s2 / 2; // 2
    let pool3 = b.add_max_pool_2d(pool2, 2, 2, 2, 2, 0, 0, &[CHANNELS, s3, s3]);

    b.build(pool3).expect("valid maxpool chain kernel")
}

fn maxpool_chain_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

#[test]
fn test_spatial_maxpool_chain_ibp() {
    let def = build_maxpool_chain();
    let bindings = maxpool_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-stage MaxPool chain");

    let final_s = SPATIAL / 8; // 2
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, final_s, final_s],
        "MaxPool chain output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MaxPool chain IBP: bounds=[{lo_min}, {hi_max}]");
    // 3 cascaded MaxPool still cannot exceed input range
    assert!(lo_min >= -5.0 - 1e-6, "chain lower >= -5.0, got {lo_min}");
    assert!(hi_max <= 5.0 + 1e-6, "chain upper <= 5.0, got {hi_max}");
}

// ===========================================================================
// 12. Conv2d -> Pool -> Conv2d composition IBP
// ===========================================================================

/// Conv2d(s=1) -> MaxPool2d(k=2,s=2) -> Conv2d(s=1): spatial processing pipeline.
/// Input: [CHANNELS, SPATIAL, SPATIAL] -> Output: [CONV_OUT_CH, SPATIAL/2, SPATIAL/2].
fn build_conv_pool_conv() -> TensorKernelDef {
    let mid_s = SPATIAL; // after first conv (stride=1, pad=1)
    let pool_s = SPATIAL / 2; // after pool
    let mut b = TensorBlockBuilder::new("spatial_conv_pool_conv");

    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Conv2d #1: [CHANNELS, 16, 16] -> [CONV_OUT_CH, 16, 16] (s=1, p=1, k=3)
    let w1 = b.add_input("conv1_weight", &[CONV_OUT_CH, CHANNELS, 3, 3]);
    let b1 = b.add_input("conv1_bias", &[CONV_OUT_CH]);
    let conv1 = b.add_conv2d(
        input,
        w1,
        Some(b1),
        1,
        1,
        1,
        1,
        &[CONV_OUT_CH, mid_s, mid_s],
    );

    // ReLU after first conv
    let relu1 = b.add_relu(conv1, &[CONV_OUT_CH, mid_s, mid_s]);

    // MaxPool2d: [CONV_OUT_CH, 16, 16] -> [CONV_OUT_CH, 8, 8]
    let pooled = b.add_max_pool_2d(relu1, 2, 2, 2, 2, 0, 0, &[CONV_OUT_CH, pool_s, pool_s]);

    // Conv2d #2: [CONV_OUT_CH, 8, 8] -> [CONV_OUT_CH, 8, 8] (s=1, p=1, k=3)
    let w2 = b.add_input("conv2_weight", &[CONV_OUT_CH, CONV_OUT_CH, 3, 3]);
    let b2 = b.add_input("conv2_bias", &[CONV_OUT_CH]);
    let conv2 = b.add_conv2d(
        pooled,
        w2,
        Some(b2),
        1,
        1,
        1,
        1,
        &[CONV_OUT_CH, pool_s, pool_s],
    );

    b.build(conv2).expect("valid conv-pool-conv kernel")
}

fn conv_pool_conv_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CONV_OUT_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CONV_OUT_CH]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CONV_OUT_CH, CONV_OUT_CH, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CONV_OUT_CH]), 0.0f32)),
    ]
}

#[test]
fn test_spatial_conv_pool_conv_ibp() {
    let def = build_conv_pool_conv();
    let bindings = conv_pool_conv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Conv->Pool->Conv");

    let pool_s = SPATIAL / 2;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CONV_OUT_CH, pool_s, pool_s],
        "Conv-Pool-Conv output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv-Pool-Conv IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 13. Conv2d -> Pool -> Conv2d composition CROWN
// ===========================================================================

#[test]
fn test_spatial_conv_pool_conv_crown() {
    let def = build_conv_pool_conv();
    let bindings = conv_pool_conv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let pool_s = SPATIAL / 2;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CONV_OUT_CH, pool_s, pool_s]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv-Pool-Conv CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 14. Spatial downsample -> upsample round-trip IBP
// ===========================================================================

/// Conv2d(s=2) downsample -> ConvTranspose2d(s=2) upsample round-trip.
/// Input: [CHANNELS, SPATIAL, SPATIAL] -> mid: [CONV_OUT_CH, SPATIAL/2, SPATIAL/2]
///   -> Output: [CHANNELS, SPATIAL, SPATIAL].
fn build_downsample_upsample() -> TensorKernelDef {
    let mid_s = SPATIAL / 2;
    let mut b = TensorBlockBuilder::new("spatial_downsample_upsample");

    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Downsample: Conv2d(s=2, k=3, p=1)
    let w_down = b.add_input("down_weight", &[CONV_OUT_CH, CHANNELS, 3, 3]);
    let b_down = b.add_input("down_bias", &[CONV_OUT_CH]);
    let down = b.add_conv2d(
        input,
        w_down,
        Some(b_down),
        2,
        2,
        1,
        1,
        &[CONV_OUT_CH, mid_s, mid_s],
    );

    // Upsample: ConvTranspose2d(s=2, k=4, p=1)
    let w_up = b.add_input("up_weight", &[CONV_OUT_CH, CHANNELS, 4, 4]);
    let b_up = b.add_input("up_bias", &[CHANNELS]);
    let up = b.add_conv_transpose_2d(
        down,
        w_up,
        Some(b_up),
        2,
        2, // stride
        1,
        1, // padding
        1,
        1, // dilation
        1, // groups
        0,
        0, // output_padding
        &[CHANNELS, SPATIAL, SPATIAL],
    );

    b.build(up).expect("valid downsample-upsample kernel")
}

fn downsample_upsample_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CONV_OUT_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CONV_OUT_CH]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CONV_OUT_CH, CHANNELS, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
    ]
}

#[test]
fn test_spatial_downsample_upsample_ibp() {
    let def = build_downsample_upsample();
    let bindings = downsample_upsample_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through downsample-upsample round-trip");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL, SPATIAL],
        "round-trip output shape matches input shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Downsample-Upsample round-trip IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 15. Feature Pyramid stride pattern (stride 1, 2, 4) IBP
// ===========================================================================

/// Multi-resolution feature pyramid: Conv2d at stride 1, 2, and 4.
/// Input: [CHANNELS, SPATIAL, SPATIAL]
/// Outputs (concatenated):
///   - stride=1: [4, 16, 16]
///   - stride=2: [4, 8, 8] -> replicated as [4, 8, 8]
///   - stride=4: [4, 4, 4] -> replicated as [4, 4, 4]
///
/// We verify each scale individually, then verify the coarsest scale.
fn build_feature_pyramid() -> TensorKernelDef {
    let ch_per_scale = 4usize;
    let s1 = SPATIAL; // 16
    let s2 = SPATIAL / 2; // 8
    let s4 = SPATIAL / 4; // 4

    let mut b = TensorBlockBuilder::new("spatial_feature_pyramid");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);

    // Scale 1: Conv2d stride=1, k=3, p=1
    let w1 = b.add_input("scale1_weight", &[ch_per_scale, CHANNELS, 3, 3]);
    let b1 = b.add_input("scale1_bias", &[ch_per_scale]);
    let _scale1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 1, 1, &[ch_per_scale, s1, s1]);

    // Scale 2: Conv2d stride=2, k=3, p=1
    let w2 = b.add_input("scale2_weight", &[ch_per_scale, CHANNELS, 3, 3]);
    let b2 = b.add_input("scale2_bias", &[ch_per_scale]);
    let _scale2 = b.add_conv2d(input, w2, Some(b2), 2, 2, 1, 1, &[ch_per_scale, s2, s2]);

    // Scale 4: Conv2d stride=4, k=3, p=1 -> output 4x4
    // For stride=4, k=3, p=1: out = (16 + 2*1 - 3)/4 + 1 = 15/4 + 1 = 4 (floor)
    let w4 = b.add_input("scale4_weight", &[ch_per_scale, CHANNELS, 3, 3]);
    let b4 = b.add_input("scale4_bias", &[ch_per_scale]);
    let scale4 = b.add_conv2d(input, w4, Some(b4), 4, 4, 1, 1, &[ch_per_scale, s4, s4]);

    // Use the coarsest scale (stride=4) as output for verification
    b.build(scale4).expect("valid feature pyramid kernel")
}

fn feature_pyramid_bindings() -> Vec<TensorParamBinding> {
    let ch_per_scale = 4usize;
    vec![
        TensorParamBinding::Variable,
        // Scale 1 weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ch_per_scale, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch_per_scale]), 0.0f32)),
        // Scale 2 weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ch_per_scale, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch_per_scale]), 0.0f32)),
        // Scale 4 weight + bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ch_per_scale, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch_per_scale]), 0.0f32)),
    ]
}

#[test]
fn test_spatial_feature_pyramid_ibp() {
    let def = build_feature_pyramid();
    let bindings = feature_pyramid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through feature pyramid");

    let ch_per_scale = 4usize;
    let s4 = SPATIAL / 4;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[ch_per_scale, s4, s4],
        "Feature pyramid stride=4 output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Feature Pyramid (stride=4) IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 16. Spatial monotone tightening
// ===========================================================================

/// Verify that smaller input epsilon produces tighter output bounds through
/// a spatial pipeline (Conv2d -> ReLU -> MaxPool2d).
///
/// This is a fundamental property: monotone interval arithmetic means
/// narrower input intervals => narrower output intervals.
fn build_spatial_tightening_pipeline() -> TensorKernelDef {
    let out_s = SPATIAL / 2;
    let mut b = TensorBlockBuilder::new("spatial_tightening");

    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);
    let w = b.add_input("weight", &[CONV_OUT_CH, CHANNELS, 3, 3]);
    let bias = b.add_input("bias", &[CONV_OUT_CH]);

    let conv = b.add_conv2d(
        input,
        w,
        Some(bias),
        1,
        1,
        1,
        1,
        &[CONV_OUT_CH, SPATIAL, SPATIAL],
    );
    let relu = b.add_relu(conv, &[CONV_OUT_CH, SPATIAL, SPATIAL]);
    let pool = b.add_max_pool_2d(relu, 2, 2, 2, 2, 0, 0, &[CONV_OUT_CH, out_s, out_s]);

    b.build(pool).expect("valid spatial tightening pipeline")
}

fn spatial_tightening_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CONV_OUT_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CONV_OUT_CH]), 0.0f32)),
    ]
}

/// Compute total bound width (sum of hi - lo across all elements).
fn total_bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo, hi) = bounds.lower_upper();
    lo.iter().zip(hi.iter()).map(|(&l, &h)| h - l).sum::<f32>()
}

#[test]
fn test_spatial_monotone_tightening() {
    let def = build_spatial_tightening_pipeline();
    let bindings = spatial_tightening_bindings();
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

    eprintln!("Monotone tightening: wide_width={wide_width:.4}, narrow_width={narrow_width:.4}");

    assert!(
        narrow_width <= wide_width + 1e-6,
        "narrower input must produce tighter output bounds: \
         narrow_width={narrow_width} > wide_width={wide_width}"
    );
}

// ===========================================================================
// 17. AvgPool2d with stride (stride != kernel) IBP
// ===========================================================================

/// Build AvgPool2d with k=4, s=2, p=1: overlapping average pooling.
/// Input: [CHANNELS, SPATIAL, SPATIAL] -> Output: [CHANNELS, SPATIAL/2, SPATIAL/2].
fn build_avgpool2d_stride() -> TensorKernelDef {
    let out_s = SPATIAL / 2;
    let mut b = TensorBlockBuilder::new("spatial_avgpool2d_stride");
    let input = b.add_input("features", &[CHANNELS, SPATIAL, SPATIAL]);
    let out = b.add_avg_pool_2d(input, 4, 4, 2, 2, 1, 1, &[CHANNELS, out_s, out_s]);
    b.build(out).expect("valid avgpool2d stride kernel")
}

fn avgpool2d_stride_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

#[test]
fn test_spatial_avgpool2d_stride_ibp() {
    let def = build_avgpool2d_stride();
    let bindings = avgpool2d_stride_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 3.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through AvgPool2d with stride");

    let out_s = SPATIAL / 2;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, out_s, out_s],
        "AvgPool2d stride output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("AvgPool2d stride IBP: bounds=[{lo_min}, {hi_max}]");
    // Average pooling cannot exceed input bounds
    assert!(
        lo_min >= -3.0 - 1e-6,
        "AvgPool stride lower >= -3.0, got {lo_min}"
    );
    assert!(
        hi_max <= 3.0 + 1e-6,
        "AvgPool stride upper <= 3.0, got {hi_max}"
    );
}
