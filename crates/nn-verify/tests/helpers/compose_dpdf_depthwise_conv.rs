// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for depthwise separable convolution bounds (MobileNet patterns).
//!
//! Verifies IBP and CROWN bound propagation through depthwise separable
//! convolution architectures used in MobileNet/EfficientNet-style document
//! understanding models. Depthwise separable convolution factorizes a standard
//! convolution into a depthwise conv (per-channel spatial filtering) followed
//! by a pointwise conv (1x1 channel mixing), reducing parameters and FLOPs.
//!
//! 1.  **Depthwise Conv2d (groups=channels) IBP**: Per-channel 3x3 spatial filtering.
//! 2.  **Pointwise Conv2d (1x1) IBP**: Channel mixing via 1x1 convolution.
//! 3.  **Depthwise separable: depthwise -> pointwise composition (IBP)**.
//! 4.  **Depthwise + BatchNorm + ReLU pipeline (IBP)**.
//! 5.  **Inverted residual (MBConv): expand -> depthwise -> project + skip (IBP)**.
//! 6.  **Squeeze-and-Excitation after depthwise conv (IBP)**.
//! 7.  **MBConv with expansion ratio 4 (IBP)**.
//! 8.  **MBConv with expansion ratio 6 (IBP)**.
//! 9.  **Stride-2 depthwise for spatial downsampling (IBP)**.
//! 10. **Depthwise conv with different kernel sizes (3x3, 5x5) (IBP)**.
//! 11. **CROWN tightness for depthwise separable blocks (CROWN)**.
//! 12. **Stacked MBConv blocks (2-layer) (IBP + CROWN)**.
//! 13. **Depthwise separable monotone tightening (IBP)**.
//! 14. **Depthwise separable vs standard conv bound comparison (IBP)**.
//! 15. **Full MBConv stage: 3 MBConv blocks with stride (IBP)**.
//!
//! Architecture references:
//! - MobileNetV2 (Sandler et al., 2018): Inverted residuals with linear bottlenecks
//! - EfficientNet (Tan & Le, 2019): Compound scaling with MBConv blocks
//! - MobileNetV3 (Howard et al., 2019): SE blocks + h-swish activation
//! - Depthwise separable convolutions (Chollet, 2017): Xception architecture
//!
//! Dimensions (small for fast verification, structurally representative):
//! - Spatial: 8x8 feature maps, 4x4 after stride-2
//! - Channels: 16 base, expansion to 64/96
//!
//! Part of #4028: Compose tests for depthwise separable convolution bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Spatial size of input feature maps.
const SPATIAL: usize = 8;
/// Spatial size after stride-2 downsampling.
const SPATIAL_HALF: usize = 4;
/// Base channel count.
const CHANNELS: usize = 16;
/// Expanded channel count (expansion ratio 4).
const EXPANDED_4X: usize = 64;
/// Expanded channel count (expansion ratio 6).
const EXPANDED_6X: usize = 96;
/// Output channel count for pointwise projection.
const OUT_CHANNELS: usize = 32;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Add a depthwise Conv2d (groups=channels) to the builder.
///
/// Weight shape: `[C, 1, kH, kW]` (one filter per channel).
/// Output shape: `[C, out_h, out_w]`.
fn add_depthwise_conv2d(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    prefix: &str,
    channels: usize,
    kernel_size: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
) -> TensorNodeId {
    let padding = kernel_size / 2;
    let w = b.add_input(
        &format!("{prefix}_dw_w"),
        &[channels, 1, kernel_size, kernel_size],
    );
    let bias = b.add_input(&format!("{prefix}_dw_b"), &[channels]);
    b.add_conv2d_full(
        input,
        w,
        Some(bias),
        stride,
        stride,
        padding,
        padding,
        1,
        1,
        channels, // groups = channels for depthwise
        &[channels, out_h, out_w],
    )
}

/// Push bindings for a depthwise Conv2d (weight + bias).
fn push_depthwise_conv_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    channels: usize,
    kernel_size: usize,
) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[channels, 1, kernel_size, kernel_size]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[channels]),
        0.0f32,
    )));
}

/// Add a pointwise Conv2d (1x1) to the builder.
///
/// Weight shape: `[C_out, C_in, 1, 1]`.
/// Output shape: `[C_out, H, W]`.
fn add_pointwise_conv2d(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    h: usize,
    w: usize,
) -> TensorNodeId {
    let weight = b.add_input(
        &format!("{prefix}_pw_w"),
        &[out_channels, in_channels, 1, 1],
    );
    let bias = b.add_input(&format!("{prefix}_pw_b"), &[out_channels]);
    b.add_conv2d(input, weight, Some(bias), 1, 1, 0, 0, &[out_channels, h, w])
}

/// Push bindings for a pointwise Conv2d (weight + bias).
fn push_pointwise_conv_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    in_channels: usize,
    out_channels: usize,
) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_channels, in_channels, 1, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_channels]),
        0.0f32,
    )));
}

/// Add a BatchNorm + ReLU block after a convolution.
fn add_bn_relu(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    prefix: &str,
    channels: usize,
    h: usize,
    w: usize,
) -> TensorNodeId {
    let shape = [channels, h, w];
    let mean = b.add_input(&format!("{prefix}_bn_mean"), &[channels]);
    let var = b.add_input(&format!("{prefix}_bn_var"), &[channels]);
    let weight = b.add_input(&format!("{prefix}_bn_weight"), &[channels]);
    let bias = b.add_input(&format!("{prefix}_bn_bias"), &[channels]);
    let eps = b.add_input(&format!("{prefix}_bn_eps"), &[1]);

    let normed = b.add_batch_norm(input, mean, var, weight, bias, eps, &shape);
    b.add_relu(normed, &shape)
}

/// Push bindings for a BatchNorm (mean, var, weight, bias, eps).
fn push_bn_bindings(bindings: &mut Vec<TensorParamBinding>, channels: usize) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[channels]),
        0.0f32,
    ))); // mean
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[channels]),
        1.0f32,
    ))); // var
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[channels]),
        1.0f32,
    ))); // weight
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[channels]),
        0.0f32,
    ))); // bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
}

/// Build an MBConv (inverted residual) block.
///
/// Pattern: expand (1x1) -> BN+ReLU -> depthwise (kxk) -> BN+ReLU -> project (1x1) -> BN + skip
///
/// Returns (kernel_def, bindings).
fn build_mbconv_block(
    name: &str,
    in_channels: usize,
    expanded_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    spatial_in: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let spatial_out = spatial_in / stride;

    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("x", &[in_channels, spatial_in, spatial_in]);

    // Expand: 1x1 conv to expand channels
    let expanded = add_pointwise_conv2d(
        &mut b,
        input,
        "expand",
        in_channels,
        expanded_channels,
        spatial_in,
        spatial_in,
    );
    let expanded = add_bn_relu(
        &mut b,
        expanded,
        "expand",
        expanded_channels,
        spatial_in,
        spatial_in,
    );

    // Depthwise: kxk conv with groups=expanded_channels
    let dw = add_depthwise_conv2d(
        &mut b,
        expanded,
        "dw",
        expanded_channels,
        kernel_size,
        stride,
        spatial_out,
        spatial_out,
    );
    let dw = add_bn_relu(
        &mut b,
        dw,
        "dw",
        expanded_channels,
        spatial_out,
        spatial_out,
    );

    // Project: 1x1 conv to reduce channels (linear bottleneck, no ReLU)
    let projected = add_pointwise_conv2d(
        &mut b,
        dw,
        "proj",
        expanded_channels,
        out_channels,
        spatial_out,
        spatial_out,
    );

    // BatchNorm on projection (no ReLU -- linear bottleneck)
    let proj_shape = [out_channels, spatial_out, spatial_out];
    let proj_mean = b.add_input("proj_bn_mean", &[out_channels]);
    let proj_var = b.add_input("proj_bn_var", &[out_channels]);
    let proj_weight = b.add_input("proj_bn_weight", &[out_channels]);
    let proj_bias = b.add_input("proj_bn_bias", &[out_channels]);
    let proj_eps = b.add_input("proj_bn_eps", &[1]);
    let projected_normed = b.add_batch_norm(
        projected,
        proj_mean,
        proj_var,
        proj_weight,
        proj_bias,
        proj_eps,
        &proj_shape,
    );

    // Skip connection only when stride=1 and in_channels==out_channels
    let out = if stride == 1 && in_channels == out_channels {
        b.add_binary_add(projected_normed, input, &proj_shape)
    } else {
        projected_normed
    };

    let def = b.build(out).expect("valid MBConv kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    // Expand conv + BN
    push_pointwise_conv_bindings(&mut bindings, in_channels, expanded_channels);
    push_bn_bindings(&mut bindings, expanded_channels);
    // Depthwise conv + BN
    push_depthwise_conv_bindings(&mut bindings, expanded_channels, kernel_size);
    push_bn_bindings(&mut bindings, expanded_channels);
    // Project conv + BN
    push_pointwise_conv_bindings(&mut bindings, expanded_channels, out_channels);
    push_bn_bindings(&mut bindings, out_channels);

    (def, bindings)
}

// ===========================================================================
// 1. Depthwise Conv2d (groups=channels) IBP
// ===========================================================================

#[test]
fn test_depthwise_conv2d_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_depthwise_conv2d");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);
    let out = add_depthwise_conv2d(&mut b, input, "dw", CHANNELS, 3, 1, SPATIAL, SPATIAL);
    let def = b.build(out).expect("valid depthwise conv kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_depthwise_conv_bindings(&mut bindings, CHANNELS, 3);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Depthwise Conv2d IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Pointwise Conv2d (1x1) IBP
// ===========================================================================

#[test]
fn test_pointwise_conv2d_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_pointwise_conv2d");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);
    let out = add_pointwise_conv2d(
        &mut b,
        input,
        "pw",
        CHANNELS,
        OUT_CHANNELS,
        SPATIAL,
        SPATIAL,
    );
    let def = b.build(out).expect("valid pointwise conv kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_pointwise_conv_bindings(&mut bindings, CHANNELS, OUT_CHANNELS);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pointwise Conv2d (1x1) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. Depthwise separable: depthwise -> pointwise composition (IBP)
// ===========================================================================

fn build_depthwise_separable_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_depthwise_separable");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);
    let dw = add_depthwise_conv2d(&mut b, input, "dw", CHANNELS, 3, 1, SPATIAL, SPATIAL);
    let out = add_pointwise_conv2d(&mut b, dw, "pw", CHANNELS, OUT_CHANNELS, SPATIAL, SPATIAL);
    b.build(out).expect("valid depthwise separable kernel")
}

fn depthwise_separable_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_depthwise_conv_bindings(&mut bindings, CHANNELS, 3);
    push_pointwise_conv_bindings(&mut bindings, CHANNELS, OUT_CHANNELS);
    bindings
}

#[test]
fn test_depthwise_separable_ibp() {
    let def = build_depthwise_separable_kernel();
    let bindings = depthwise_separable_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Depthwise separable (DW+PW) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 4. Depthwise + BatchNorm + ReLU pipeline (IBP)
// ===========================================================================

#[test]
fn test_depthwise_bn_relu_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_depthwise_bn_relu");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);
    let dw = add_depthwise_conv2d(&mut b, input, "dw", CHANNELS, 3, 1, SPATIAL, SPATIAL);
    let out = add_bn_relu(&mut b, dw, "dw", CHANNELS, SPATIAL, SPATIAL);
    let def = b.build(out).expect("valid depthwise BN ReLU kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_depthwise_conv_bindings(&mut bindings, CHANNELS, 3);
    push_bn_bindings(&mut bindings, CHANNELS);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Depthwise + BN + ReLU IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // ReLU ensures non-negative lower bound
    assert!(
        lo_min >= -1e-6,
        "ReLU output lower must be >= 0, got {lo_min}"
    );
}

// ===========================================================================
// 5. Inverted residual (MBConv): expand -> depthwise -> project + skip (IBP)
// ===========================================================================

#[test]
fn test_mbconv_inverted_residual_ibp() {
    let (def, bindings) = build_mbconv_block(
        "dpdf_mbconv_basic",
        CHANNELS,
        EXPANDED_4X,
        CHANNELS, // same in/out for skip connection
        3,
        1,
        SPATIAL,
    );

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MBConv inverted residual IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. Squeeze-and-Excitation after depthwise conv (IBP)
// ===========================================================================

/// Build SE block: global avg pool -> FC -> ReLU -> FC -> sigmoid -> scale.
///
/// Input: [C, H, W]. Output: [C, H, W] (channel-recalibrated).
fn build_se_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    prefix: &str,
    channels: usize,
    h: usize,
    w: usize,
) -> TensorNodeId {
    let se_reduce = channels / 4; // Reduction ratio 4
    let spatial_shape = [channels, h, w];

    // Global average pooling: [C, H, W] -> [C, 1, 1]
    let pooled = b.add_avg_pool_2d(input, h, w, h, w, 0, 0, &[channels, 1, 1]);

    // FC reduce: [C, 1, 1] linear approx via 1x1 conv
    let fc1_w = b.add_input(&format!("{prefix}_se_fc1_w"), &[se_reduce, channels, 1, 1]);
    let fc1_b = b.add_input(&format!("{prefix}_se_fc1_b"), &[se_reduce]);
    let fc1_out = b.add_conv2d(pooled, fc1_w, Some(fc1_b), 1, 1, 0, 0, &[se_reduce, 1, 1]);
    let fc1_relu = b.add_relu(fc1_out, &[se_reduce, 1, 1]);

    // FC expand: [se_reduce, 1, 1] -> [C, 1, 1]
    let fc2_w = b.add_input(&format!("{prefix}_se_fc2_w"), &[channels, se_reduce, 1, 1]);
    let fc2_b = b.add_input(&format!("{prefix}_se_fc2_b"), &[channels]);
    let fc2_out = b.add_conv2d(fc1_relu, fc2_w, Some(fc2_b), 1, 1, 0, 0, &[channels, 1, 1]);
    let gate = b.add_sigmoid(fc2_out, &[channels, 1, 1]);

    // Broadcast gate to spatial dims and multiply
    let gate_bc = b.add_broadcast(gate, &spatial_shape);
    b.add_binary_mul(input, gate_bc, &spatial_shape)
}

/// Push bindings for an SE block (fc1_w, fc1_b, fc2_w, fc2_b).
fn push_se_bindings(bindings: &mut Vec<TensorParamBinding>, channels: usize) {
    let se_reduce = channels / 4;
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[se_reduce, channels, 1, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[se_reduce]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[channels, se_reduce, 1, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[channels]),
        0.0f32,
    )));
}

#[test]
fn test_se_after_depthwise_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_se_depthwise");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);

    // Depthwise conv -> SE block
    let dw = add_depthwise_conv2d(&mut b, input, "dw", CHANNELS, 3, 1, SPATIAL, SPATIAL);
    let out = build_se_block(&mut b, dw, "se", CHANNELS, SPATIAL, SPATIAL);
    let def = b.build(out).expect("valid SE depthwise kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_depthwise_conv_bindings(&mut bindings, CHANNELS, 3);
    push_se_bindings(&mut bindings, CHANNELS);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SE + depthwise conv IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. MBConv with expansion ratio 4 (IBP)
// ===========================================================================

#[test]
fn test_mbconv_expansion_ratio_4_ibp() {
    let (def, bindings) = build_mbconv_block(
        "dpdf_mbconv_exp4",
        CHANNELS,
        EXPANDED_4X, // 16 * 4 = 64
        CHANNELS,
        3,
        1,
        SPATIAL,
    );

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("MBConv expansion=4 IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

// ===========================================================================
// 8. MBConv with expansion ratio 6 (IBP)
// ===========================================================================

#[test]
fn test_mbconv_expansion_ratio_6_ibp() {
    let (def, bindings) = build_mbconv_block(
        "dpdf_mbconv_exp6",
        CHANNELS,
        EXPANDED_6X, // 16 * 6 = 96
        CHANNELS,
        3,
        1,
        SPATIAL,
    );

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("MBConv expansion=6 IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

// ===========================================================================
// 9. Stride-2 depthwise for spatial downsampling (IBP)
// ===========================================================================

#[test]
fn test_depthwise_stride2_downsample_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_depthwise_stride2");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);
    let dw = add_depthwise_conv2d(
        &mut b,
        input,
        "dw",
        CHANNELS,
        3,
        2,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let out = add_bn_relu(&mut b, dw, "dw", CHANNELS, SPATIAL_HALF, SPATIAL_HALF);
    let def = b.build(out).expect("valid stride-2 depthwise kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_depthwise_conv_bindings(&mut bindings, CHANNELS, 3);
    push_bn_bindings(&mut bindings, CHANNELS);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    // Verify output spatial dimensions are halved
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL_HALF, SPATIAL_HALF]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Stride-2 depthwise IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min >= -1e-6,
        "ReLU output lower must be >= 0, got {lo_min}"
    );
}

// ===========================================================================
// 10. Depthwise conv with different kernel sizes (3x3, 5x5) (IBP)
// ===========================================================================

fn test_depthwise_kernel_size(kernel_size: usize) -> f32 {
    let mut b = TensorBlockBuilder::new(&format!("dpdf_dw_k{kernel_size}"));
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);
    let out = add_depthwise_conv2d(
        &mut b,
        input,
        "dw",
        CHANNELS,
        kernel_size,
        1,
        SPATIAL,
        SPATIAL,
    );
    let def = b.build(out).expect("valid depthwise kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_depthwise_conv_bindings(&mut bindings, CHANNELS, kernel_size);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    bound_width(&output)
}

#[test]
fn test_depthwise_kernel_3x3_vs_5x5_ibp() {
    let width_3x3 = test_depthwise_kernel_size(3);
    let width_5x5 = test_depthwise_kernel_size(5);

    eprintln!("Depthwise kernel sizes IBP: 3x3 width={width_3x3:.6}, 5x5 width={width_5x5:.6}");
    assert!(width_3x3.is_finite(), "3x3 width must be finite");
    assert!(width_5x5.is_finite(), "5x5 width must be finite");
    // Larger kernel touches more inputs, so may produce wider bounds
    // (not strictly guaranteed, but both must be finite)
}

// ===========================================================================
// 11. CROWN tightness for depthwise separable blocks (CROWN)
// ===========================================================================

#[test]
fn test_depthwise_separable_crown() {
    let def = build_depthwise_separable_kernel();
    let bindings = depthwise_separable_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Depthwise separable CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 12. Stacked MBConv blocks (2-layer) (IBP + CROWN)
// ===========================================================================

fn build_stacked_mbconv_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("dpdf_stacked_mbconv_2");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);

    // Block 1: expand -> DW -> project + skip
    let expand1 = add_pointwise_conv2d(
        &mut b,
        input,
        "b1_expand",
        CHANNELS,
        EXPANDED_4X,
        SPATIAL,
        SPATIAL,
    );
    let expand1 = add_bn_relu(&mut b, expand1, "b1_expand", EXPANDED_4X, SPATIAL, SPATIAL);
    let dw1 = add_depthwise_conv2d(
        &mut b,
        expand1,
        "b1_dw",
        EXPANDED_4X,
        3,
        1,
        SPATIAL,
        SPATIAL,
    );
    let dw1 = add_bn_relu(&mut b, dw1, "b1_dw", EXPANDED_4X, SPATIAL, SPATIAL);
    let proj1 = add_pointwise_conv2d(
        &mut b,
        dw1,
        "b1_proj",
        EXPANDED_4X,
        CHANNELS,
        SPATIAL,
        SPATIAL,
    );
    let shape = [CHANNELS, SPATIAL, SPATIAL];
    let proj1_mean = b.add_input("b1_proj_bn_mean", &[CHANNELS]);
    let proj1_var = b.add_input("b1_proj_bn_var", &[CHANNELS]);
    let proj1_w = b.add_input("b1_proj_bn_weight", &[CHANNELS]);
    let proj1_bias = b.add_input("b1_proj_bn_bias", &[CHANNELS]);
    let proj1_eps = b.add_input("b1_proj_bn_eps", &[1]);
    let proj1_normed = b.add_batch_norm(
        proj1, proj1_mean, proj1_var, proj1_w, proj1_bias, proj1_eps, &shape,
    );
    let h1 = b.add_binary_add(proj1_normed, input, &shape);

    // Block 2: expand -> DW -> project + skip
    let expand2 = add_pointwise_conv2d(
        &mut b,
        h1,
        "b2_expand",
        CHANNELS,
        EXPANDED_4X,
        SPATIAL,
        SPATIAL,
    );
    let expand2 = add_bn_relu(&mut b, expand2, "b2_expand", EXPANDED_4X, SPATIAL, SPATIAL);
    let dw2 = add_depthwise_conv2d(
        &mut b,
        expand2,
        "b2_dw",
        EXPANDED_4X,
        3,
        1,
        SPATIAL,
        SPATIAL,
    );
    let dw2 = add_bn_relu(&mut b, dw2, "b2_dw", EXPANDED_4X, SPATIAL, SPATIAL);
    let proj2 = add_pointwise_conv2d(
        &mut b,
        dw2,
        "b2_proj",
        EXPANDED_4X,
        CHANNELS,
        SPATIAL,
        SPATIAL,
    );
    let proj2_mean = b.add_input("b2_proj_bn_mean", &[CHANNELS]);
    let proj2_var = b.add_input("b2_proj_bn_var", &[CHANNELS]);
    let proj2_w = b.add_input("b2_proj_bn_weight", &[CHANNELS]);
    let proj2_bias = b.add_input("b2_proj_bn_bias", &[CHANNELS]);
    let proj2_eps = b.add_input("b2_proj_bn_eps", &[1]);
    let proj2_normed = b.add_batch_norm(
        proj2, proj2_mean, proj2_var, proj2_w, proj2_bias, proj2_eps, &shape,
    );
    let out = b.add_binary_add(proj2_normed, h1, &shape);

    let def = b.build(out).expect("valid stacked MBConv kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    // Block 1: expand conv + BN, DW conv + BN, proj conv + BN
    push_pointwise_conv_bindings(&mut bindings, CHANNELS, EXPANDED_4X);
    push_bn_bindings(&mut bindings, EXPANDED_4X);
    push_depthwise_conv_bindings(&mut bindings, EXPANDED_4X, 3);
    push_bn_bindings(&mut bindings, EXPANDED_4X);
    push_pointwise_conv_bindings(&mut bindings, EXPANDED_4X, CHANNELS);
    push_bn_bindings(&mut bindings, CHANNELS);
    // Block 2: expand conv + BN, DW conv + BN, proj conv + BN
    push_pointwise_conv_bindings(&mut bindings, CHANNELS, EXPANDED_4X);
    push_bn_bindings(&mut bindings, EXPANDED_4X);
    push_depthwise_conv_bindings(&mut bindings, EXPANDED_4X, 3);
    push_bn_bindings(&mut bindings, EXPANDED_4X);
    push_pointwise_conv_bindings(&mut bindings, EXPANDED_4X, CHANNELS);
    push_bn_bindings(&mut bindings, CHANNELS);

    (def, bindings)
}

#[test]
fn test_stacked_mbconv_2_ibp() {
    let (def, bindings) = build_stacked_mbconv_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Stacked MBConv (2-layer) IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

#[test]
fn test_stacked_mbconv_2_crown() {
    let (def, bindings) = build_stacked_mbconv_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("Stacked MBConv (2-layer) CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 13. Depthwise separable monotone tightening (IBP)
// ===========================================================================

#[test]
fn test_depthwise_separable_monotone_tightening_ibp() {
    let def = build_depthwise_separable_kernel();
    let bindings = depthwise_separable_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);
    let wide_width = bound_width(&wide_output);

    let tight_input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 0.1);
    let tight_output = graph.propagate_ibp(&tight_input).expect("IBP tight");
    assert_bounds_valid(&tight_output);
    let tight_width = bound_width(&tight_output);

    eprintln!(
        "Depthwise separable monotone: eps=1.0 width={wide_width:.6}, eps=0.1 width={tight_width:.6}"
    );
    assert!(
        tight_width <= wide_width + 1e-6,
        "tight input should produce tighter output: wide={wide_width}, tight={tight_width}"
    );
}

// ===========================================================================
// 14. Depthwise separable vs standard conv bound comparison (IBP)
// ===========================================================================

#[test]
fn test_depthwise_separable_vs_standard_conv_ibp() {
    // Depthwise separable: DW(3x3) + PW(1x1)
    let ds_def = build_depthwise_separable_kernel();
    let ds_bindings = depthwise_separable_bindings();
    let ds_graph = tensor_kernel_to_graph(&ds_def, &ds_bindings).expect("DS graph");

    // Standard conv: Conv2d(in_c, out_c, 3x3)
    let mut b = TensorBlockBuilder::new("dpdf_standard_conv");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);
    let w = b.add_input("conv_w", &[OUT_CHANNELS, CHANNELS, 3, 3]);
    let bias = b.add_input("conv_b", &[OUT_CHANNELS]);
    let out = b.add_conv2d(
        input,
        w,
        Some(bias),
        1,
        1,
        1,
        1,
        &[OUT_CHANNELS, SPATIAL, SPATIAL],
    );
    let std_def = b.build(out).expect("valid standard conv kernel");

    let std_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OUT_CHANNELS, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[OUT_CHANNELS]), 0.0f32)),
    ];
    let std_graph = tensor_kernel_to_graph(&std_def, &std_bindings).expect("std graph");

    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let ds_output = ds_graph.propagate_ibp(&input).expect("DS IBP");
    let std_output = std_graph.propagate_ibp(&input).expect("standard IBP");

    assert_bounds_valid(&ds_output);
    assert_bounds_valid(&std_output);

    let ds_width = bound_width(&ds_output);
    let std_width = bound_width(&std_output);
    eprintln!(
        "DW-separable vs standard conv IBP: ds_width={ds_width:.6}, std_width={std_width:.6}"
    );
    // Both must produce finite bounds; depthwise separable has fewer parameters
    assert!(ds_width.is_finite(), "DS width must be finite");
    assert!(std_width.is_finite(), "standard conv width must be finite");
}

// ===========================================================================
// 15. Full MBConv stage: 3 MBConv blocks with stride (IBP)
// ===========================================================================

#[test]
fn test_full_mbconv_stage_3_blocks_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mbconv_stage_3");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);

    // Block 1: stride-2 downsampling, CHANNELS -> OUT_CHANNELS
    let expand1 = add_pointwise_conv2d(
        &mut b,
        input,
        "s1_expand",
        CHANNELS,
        EXPANDED_4X,
        SPATIAL,
        SPATIAL,
    );
    let expand1 = add_bn_relu(&mut b, expand1, "s1_expand", EXPANDED_4X, SPATIAL, SPATIAL);
    let dw1 = add_depthwise_conv2d(
        &mut b,
        expand1,
        "s1_dw",
        EXPANDED_4X,
        3,
        2,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let dw1 = add_bn_relu(
        &mut b,
        dw1,
        "s1_dw",
        EXPANDED_4X,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let proj1 = add_pointwise_conv2d(
        &mut b,
        dw1,
        "s1_proj",
        EXPANDED_4X,
        OUT_CHANNELS,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let proj1_shape = [OUT_CHANNELS, SPATIAL_HALF, SPATIAL_HALF];
    let p1_mean = b.add_input("s1_proj_bn_mean", &[OUT_CHANNELS]);
    let p1_var = b.add_input("s1_proj_bn_var", &[OUT_CHANNELS]);
    let p1_w = b.add_input("s1_proj_bn_weight", &[OUT_CHANNELS]);
    let p1_bias = b.add_input("s1_proj_bn_bias", &[OUT_CHANNELS]);
    let p1_eps = b.add_input("s1_proj_bn_eps", &[1]);
    let h1 = b.add_batch_norm(proj1, p1_mean, p1_var, p1_w, p1_bias, p1_eps, &proj1_shape);
    // No skip (stride != 1 or channels changed)

    // Block 2: stride-1, OUT_CHANNELS -> OUT_CHANNELS (with skip)
    let expand2 = add_pointwise_conv2d(
        &mut b,
        h1,
        "s2_expand",
        OUT_CHANNELS,
        EXPANDED_4X,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let expand2 = add_bn_relu(
        &mut b,
        expand2,
        "s2_expand",
        EXPANDED_4X,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let dw2 = add_depthwise_conv2d(
        &mut b,
        expand2,
        "s2_dw",
        EXPANDED_4X,
        3,
        1,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let dw2 = add_bn_relu(
        &mut b,
        dw2,
        "s2_dw",
        EXPANDED_4X,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let proj2 = add_pointwise_conv2d(
        &mut b,
        dw2,
        "s2_proj",
        EXPANDED_4X,
        OUT_CHANNELS,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let p2_mean = b.add_input("s2_proj_bn_mean", &[OUT_CHANNELS]);
    let p2_var = b.add_input("s2_proj_bn_var", &[OUT_CHANNELS]);
    let p2_w = b.add_input("s2_proj_bn_weight", &[OUT_CHANNELS]);
    let p2_bias = b.add_input("s2_proj_bn_bias", &[OUT_CHANNELS]);
    let p2_eps = b.add_input("s2_proj_bn_eps", &[1]);
    let proj2_normed =
        b.add_batch_norm(proj2, p2_mean, p2_var, p2_w, p2_bias, p2_eps, &proj1_shape);
    let h2 = b.add_binary_add(proj2_normed, h1, &proj1_shape);

    // Block 3: stride-1, OUT_CHANNELS -> OUT_CHANNELS (with skip)
    let expand3 = add_pointwise_conv2d(
        &mut b,
        h2,
        "s3_expand",
        OUT_CHANNELS,
        EXPANDED_4X,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let expand3 = add_bn_relu(
        &mut b,
        expand3,
        "s3_expand",
        EXPANDED_4X,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let dw3 = add_depthwise_conv2d(
        &mut b,
        expand3,
        "s3_dw",
        EXPANDED_4X,
        3,
        1,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let dw3 = add_bn_relu(
        &mut b,
        dw3,
        "s3_dw",
        EXPANDED_4X,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let proj3 = add_pointwise_conv2d(
        &mut b,
        dw3,
        "s3_proj",
        EXPANDED_4X,
        OUT_CHANNELS,
        SPATIAL_HALF,
        SPATIAL_HALF,
    );
    let p3_mean = b.add_input("s3_proj_bn_mean", &[OUT_CHANNELS]);
    let p3_var = b.add_input("s3_proj_bn_var", &[OUT_CHANNELS]);
    let p3_w = b.add_input("s3_proj_bn_weight", &[OUT_CHANNELS]);
    let p3_bias = b.add_input("s3_proj_bn_bias", &[OUT_CHANNELS]);
    let p3_eps = b.add_input("s3_proj_bn_eps", &[1]);
    let proj3_normed =
        b.add_batch_norm(proj3, p3_mean, p3_var, p3_w, p3_bias, p3_eps, &proj1_shape);
    let out = b.add_binary_add(proj3_normed, h2, &proj1_shape);

    let def = b.build(out).expect("valid MBConv stage kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    // Block 1: expand(CHANNELS->EXPANDED) + BN, DW + BN, proj(EXPANDED->OUT) + BN
    push_pointwise_conv_bindings(&mut bindings, CHANNELS, EXPANDED_4X);
    push_bn_bindings(&mut bindings, EXPANDED_4X);
    push_depthwise_conv_bindings(&mut bindings, EXPANDED_4X, 3);
    push_bn_bindings(&mut bindings, EXPANDED_4X);
    push_pointwise_conv_bindings(&mut bindings, EXPANDED_4X, OUT_CHANNELS);
    push_bn_bindings(&mut bindings, OUT_CHANNELS);
    // Block 2: expand(OUT->EXPANDED) + BN, DW + BN, proj(EXPANDED->OUT) + BN
    push_pointwise_conv_bindings(&mut bindings, OUT_CHANNELS, EXPANDED_4X);
    push_bn_bindings(&mut bindings, EXPANDED_4X);
    push_depthwise_conv_bindings(&mut bindings, EXPANDED_4X, 3);
    push_bn_bindings(&mut bindings, EXPANDED_4X);
    push_pointwise_conv_bindings(&mut bindings, EXPANDED_4X, OUT_CHANNELS);
    push_bn_bindings(&mut bindings, OUT_CHANNELS);
    // Block 3: expand(OUT->EXPANDED) + BN, DW + BN, proj(EXPANDED->OUT) + BN
    push_pointwise_conv_bindings(&mut bindings, OUT_CHANNELS, EXPANDED_4X);
    push_bn_bindings(&mut bindings, EXPANDED_4X);
    push_depthwise_conv_bindings(&mut bindings, EXPANDED_4X, 3);
    push_bn_bindings(&mut bindings, EXPANDED_4X);
    push_pointwise_conv_bindings(&mut bindings, EXPANDED_4X, OUT_CHANNELS);
    push_bn_bindings(&mut bindings, OUT_CHANNELS);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    // Verify output shape is [OUT_CHANNELS, SPATIAL_HALF, SPATIAL_HALF]
    assert_eq!(
        output.lower_upper().0.shape(),
        &[OUT_CHANNELS, SPATIAL_HALF, SPATIAL_HALF]
    );

    let width = bound_width(&output);
    eprintln!("Full MBConv stage (3 blocks, stride-2 first) IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}
