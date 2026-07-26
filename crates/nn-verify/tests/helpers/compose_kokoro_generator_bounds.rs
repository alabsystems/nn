// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! IBP compose tests for Kokoro Generator pipeline bounds.
//!
//! The Kokoro TTS model uses a BigVGAN-style generator that converts hidden
//! representations to audio waveforms. The generator pipeline is:
//!   1. Conv1d upsampling (transposed conv, channels decrease by 2x per stage)
//!   2. ResBlock chains with Snake activation and style injection
//!   3. Final Conv1d -> tanh activation -> audio output
//!
//! This file verifies 8 IBP properties of the generator pipeline:
//!
//! 1. **Conv1d upsample stage** -- ConvTranspose1d doubles temporal dimension.
//! 2. **Snake activation bounds** -- element-wise Snake preserves bounded output.
//! 3. **ResBlock residual add** -- input + conv_output stays bounded.
//! 4. **Style projection injection** -- style_embed -> affine -> modulates hidden.
//! 5. **Generator output tanh** -- tanh bounds must be in [-1, 1].
//! 6. **Multi-stage upsample chain** -- 512->256->128->64 channel cascade.
//! 7. **Dilated convolution bounds** -- dilation > 1 maintains finite bounds.
//! 8. **Fused ResBlock with style projection** -- combined block bounds.
//!
//! All tests use small dims (C<=16, T<=16) and IBP propagation through proxy
//! graphs built with TensorBlockBuilder.
//!
//! Part of #3351: Epic -- Absolutely Best Kokoro.

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

use super::common::{
    assert_bounds_valid, assert_bounds_width, assert_norm_spatial_non_degenerate, bounds_min_max,
    uniform_bounds,
};

// ===========================================================================
// Constants
// ===========================================================================

/// Small weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.01;

/// Initial generator channels (production: 512; toy scale).
const CH_IN: usize = 16;

/// Second stage channels (CH_IN / 2).
const CH_MID: usize = CH_IN / 2;

/// Third stage channels (CH_IN / 4).
const CH_OUT: usize = CH_IN / 4;

/// Temporal dimension (must be > 1 for InstanceNorm spatial non-degeneracy).
const T_LEN: usize = 16;

/// Kernel size for standard convolutions.
const KERNEL_SIZE: usize = 3;

/// Kernel size for upsample transposed convolutions.
///
/// Must be odd so symmetric same-padding `(K-1)/2` preserves the temporal
/// dimension exactly under a stride-1 Conv1d proxy: with an even kernel,
/// `(K-1)/2` truncates and the output is one shorter than the input,
/// which would mis-declare the `[C_out, T]` output shape.
const UPSAMPLE_KERNEL: usize = 7;

/// Style embedding dimension (production: 256; toy scale).
const STYLE_DIM: usize = 8;

/// Vacuous width threshold -- bounds wider than this are meaningless for
/// individual generator components.
const VACUOUS_THRESHOLD: f32 = 500.0;

// ===========================================================================
// Builder helpers
// ===========================================================================

/// Compute padding to preserve temporal dimension for a dilated conv1d.
///
/// For kernel_size=3 with dilation d: effective_kernel = 1 + d * (kernel_size - 1)
/// padding = (effective_kernel - 1) / 2 = d * (kernel_size - 1) / 2 = d (for k=3).
fn dilated_padding(kernel_size: usize, dilation: usize) -> usize {
    dilation * (kernel_size - 1) / 2
}

/// Build a Conv1d upsample stage proxy graph.
///
/// In the real generator, this is a ConvTranspose1d that doubles the temporal
/// dimension. For IBP verification, we model this as a standard Conv1d
/// (stride=1) that maps channels_in -> channels_out, since the key property
/// is channel reduction and bound propagation through convolutional weights.
///
/// Input: `[C_in, T]` (Variable).
/// Output: `[C_out, T]`.
fn build_upsample_conv(
    channels_in: usize,
    channels_out: usize,
    time_len: usize,
) -> TensorKernelDef {
    let in_shape = [channels_in, time_len];
    let out_shape = [channels_out, time_len];
    let padding = (UPSAMPLE_KERNEL - 1) / 2; // same-padding approximation
    let mut b = TensorBlockBuilder::new("generator_upsample_conv");

    let x = b.add_input("x", &in_shape);
    let w = b.add_input("w", &[channels_out, channels_in, UPSAMPLE_KERNEL]);
    let bias = b.add_input("bias", &[channels_out]);
    let out = b.add_conv1d(x, w, Some(bias), 1, padding, &out_shape);

    b.build(out).expect("valid upsample conv graph")
}

/// Bindings for upsample conv.
fn upsample_conv_bindings(channels_in: usize, channels_out: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[channels_out, channels_in, UPSAMPLE_KERNEL]),
            WEIGHT_MAG,
        )), // w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels_out]), 0.0f32)), // bias
    ]
}

/// Build a Snake activation proxy graph: `x + (1/alpha) * sin(alpha*x)^2`.
///
/// Input: `[C, T]` (Variable).
/// Output: `[C, T]`.
fn build_snake_activation(channels: usize, time_len: usize) -> TensorKernelDef {
    let shape = [channels, time_len];
    let mut b = TensorBlockBuilder::new("generator_snake_activation");

    let x = b.add_input("x", &shape);
    let alpha = b.add_input("alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &shape);
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let out = b.add_elementwise(snake_kernel, &[x, alpha_bc], &shape);

    b.build(out).expect("valid snake activation graph")
}

/// Bindings for Snake activation.
fn snake_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,            // x
        TensorParamBinding::ConstantScalar(1.0), // alpha
    ]
}

/// Build a residual add graph: `skip + conv_branch`.
///
/// Input: two Variable inputs with shape `[C, T]`.
/// Output: `[C, T]`.
fn build_residual_add(channels: usize, time_len: usize) -> TensorKernelDef {
    let shape = [channels, time_len];
    let mut b = TensorBlockBuilder::new("generator_residual_add");

    let skip = b.add_input("skip", &shape);
    let branch = b.add_input("branch", &shape);
    let out = b.add_binary_add(skip, branch, &shape);

    b.build(out).expect("valid residual add graph")
}

/// Bindings for residual addition (both inputs are Variable).
fn residual_add_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // skip
        TensorParamBinding::Variable, // branch
    ]
}

/// Build a style projection injection graph.
///
/// Architecture: Linear(style_dim, 2*channels) -> split -> gamma, beta
/// Then: gamma * hidden + beta (AdaIN-style modulation).
///
/// For IBP we model the linear + affine modulation as:
///   style -> Linear -> narrow(gamma) + narrow(beta)
///   hidden * broadcast(gamma) + broadcast(beta)
///
/// Input: Variable `[C, T]` (hidden state) + Variable `[STYLE_DIM]` (style).
/// Output: `[C, T]`.
fn build_style_projection(channels: usize, time_len: usize, style_dim: usize) -> TensorKernelDef {
    let hidden_shape = [channels, time_len];
    let style_shape = [style_dim];
    let proj_shape = [2 * channels]; // gamma + beta concatenated
    let gamma_shape = [channels];

    let mut b = TensorBlockBuilder::new("generator_style_projection");

    let hidden = b.add_input("hidden", &hidden_shape);
    let style = b.add_input("style", &style_shape);
    let proj_w = b.add_input("proj_w", &[2 * channels, style_dim]);
    let proj_b = b.add_input("proj_b", &proj_shape);

    // Linear: style -> [2*C]
    let proj = b.add_linear(style, proj_w, Some(proj_b), &proj_shape);

    // Split: gamma = proj[0:C], beta = proj[C:2C]
    let gamma = b.add_narrow(proj, 0, 0, channels, &gamma_shape);
    let beta = b.add_narrow(proj, 0, channels, channels, &gamma_shape);

    // Broadcast gamma/beta to match hidden [C, T]
    let gamma_bc = b.add_broadcast_left(gamma, &hidden_shape);
    let beta_bc = b.add_broadcast_left(beta, &hidden_shape);

    // AdaIN: hidden * gamma + beta
    let scaled = b.add_binary_mul(hidden, gamma_bc, &hidden_shape);
    let out = b.add_binary_add(scaled, beta_bc, &hidden_shape);

    b.build(out).expect("valid style projection graph")
}

/// Bindings for style projection.
fn style_projection_bindings(channels: usize, style_dim: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden
        TensorParamBinding::Variable, // style
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[2 * channels, style_dim]),
            WEIGHT_MAG,
        )), // proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[2 * channels]), 0.0f32)), // proj_b
    ]
}

/// Build a tanh output stage: Conv1d -> tanh.
///
/// Input: `[C, T]` (Variable).
/// Output: `[1, T]` (single-channel audio).
fn build_tanh_output(channels: usize, time_len: usize) -> TensorKernelDef {
    let in_shape = [channels, time_len];
    let out_shape = [1, time_len];
    let mut b = TensorBlockBuilder::new("generator_tanh_output");

    let x = b.add_input("x", &in_shape);
    let w = b.add_input("w", &[1, channels, KERNEL_SIZE]);
    let bias = b.add_input("bias", &[1]);
    let padding = (KERNEL_SIZE - 1) / 2;
    let conv = b.add_conv1d(x, w, Some(bias), 1, padding, &out_shape);
    let out = b.add_tanh(conv, &out_shape);

    b.build(out).expect("valid tanh output graph")
}

/// Bindings for tanh output.
fn tanh_output_bindings(channels: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[1, channels, KERNEL_SIZE]),
            WEIGHT_MAG,
        )), // w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)), // bias
    ]
}

/// Build a dilated Conv1d graph (no activation, no norm).
///
/// Input: `[C, T]` (Variable).
/// Output: `[C, T]` (same shape with appropriate padding).
fn build_dilated_conv1d(channels: usize, time_len: usize, dilation: usize) -> TensorKernelDef {
    let shape = [channels, time_len];
    let padding = dilated_padding(KERNEL_SIZE, dilation);
    let mut b = TensorBlockBuilder::new("generator_dilated_conv1d");

    let x = b.add_input("x", &shape);
    let w = b.add_input("w", &[channels, channels, KERNEL_SIZE]);
    let bias = b.add_input("bias", &[channels]);
    let out = b.add_conv1d_full(x, w, Some(bias), 1, padding, dilation, 1, &shape);

    b.build(out).expect("valid dilated conv1d graph")
}

/// Bindings for a dilated Conv1d.
fn dilated_conv_bindings(channels: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[channels, channels, KERNEL_SIZE]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)),
    ]
}

/// Build a fused ResBlock with style projection.
///
/// Architecture:
///   InstanceNorm -> Snake -> Conv1d(dilated) -> StyleProjection(AdaIN) + residual
///
/// Input: `[C, T]` (Variable) + `[STYLE_DIM]` (style, Variable).
/// Output: `[C, T]`.
fn build_fused_resblock_with_style(
    channels: usize,
    time_len: usize,
    style_dim: usize,
    dilation: usize,
) -> TensorKernelDef {
    assert_norm_spatial_non_degenerate(time_len, "fused_resblock_style");
    let shape = [channels, time_len];
    let padding = dilated_padding(KERNEL_SIZE, dilation);

    let mut b = TensorBlockBuilder::new("generator_fused_resblock_style");

    // Inputs
    let x = b.add_input("x", &shape);
    let style = b.add_input("style", &[style_dim]);
    let eps = b.add_input("eps", &[1]);

    // InstanceNorm
    let gamma = b.add_input("gamma", &[channels]);
    let beta = b.add_input("beta", &[channels]);
    let norm = b.add_instance_norm(x, eps, 1, Some(gamma), Some(beta), &shape);

    // Snake activation
    let alpha = b.add_input("alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &shape);
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let snake = b.add_elementwise(snake_kernel, &[norm, alpha_bc], &shape);

    // Conv1d (dilated)
    let conv_w = b.add_input("conv_w", &[channels, channels, KERNEL_SIZE]);
    let conv_b = b.add_input("conv_b", &[channels]);
    let conv = b.add_conv1d_full(snake, conv_w, Some(conv_b), 1, padding, dilation, 1, &shape);

    // Style projection: Linear(style) -> gamma_s, beta_s -> affine modulation
    let proj_w = b.add_input("proj_w", &[2 * channels, style_dim]);
    let proj_b = b.add_input("proj_b", &[2 * channels]);
    let proj = b.add_linear(style, proj_w, Some(proj_b), &[2 * channels]);

    let gamma_s = b.add_narrow(proj, 0, 0, channels, &[channels]);
    let beta_s = b.add_narrow(proj, 0, channels, channels, &[channels]);
    let gamma_s_bc = b.add_broadcast_left(gamma_s, &shape);
    let beta_s_bc = b.add_broadcast_left(beta_s, &shape);

    let styled = b.add_binary_mul(conv, gamma_s_bc, &shape);
    let modulated = b.add_binary_add(styled, beta_s_bc, &shape);

    // Residual: x + modulated
    let out = b.add_binary_add(x, modulated, &shape);

    b.build(out).expect("valid fused resblock with style graph")
}

/// Bindings for fused ResBlock with style projection.
fn fused_resblock_style_bindings(channels: usize, style_dim: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::Variable,             // style
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 1.0f32)), // gamma
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // beta
        TensorParamBinding::ConstantScalar(1.0),  // alpha
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[channels, channels, KERNEL_SIZE]),
            WEIGHT_MAG,
        )), // conv_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // conv_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[2 * channels, style_dim]),
            WEIGHT_MAG,
        )), // proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[2 * channels]), 0.0f32)), // proj_b
    ]
}

// ===========================================================================
// Test 1: Conv1d upsample stage preserves bounds
// ===========================================================================

/// Conv1d upsample stage preserves temporal bounds.
///
/// Architecture: Conv1d with large kernel (8) maps channels_in -> channels_out.
/// In the real generator this is a ConvTranspose1d(stride=2), but for IBP
/// verification we model it as a standard Conv1d since the key property is
/// bound propagation through the convolutional weight matrix.
///
/// IBP propagates Conv1d bounds as interval matrix-vector multiply:
/// output bounds scale proportionally with weight magnitude * input range.
#[test]
fn test_generator_upsample_conv_bounds() {
    let def = build_upsample_conv(CH_IN, CH_MID, T_LEN);
    def.validate().expect("upsample conv def validates");

    let bindings = upsample_conv_bindings(CH_IN, CH_MID);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CH_IN, T_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through upsample conv");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "upsample conv bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CH_MID, T_LEN],
        "upsample conv must produce [C_out, T] shape"
    );

    // Conv1d with small weights (0.01) and input in [-1, 1]:
    // Each output element is a weighted sum of CH_IN * UPSAMPLE_KERNEL inputs.
    // Maximum magnitude: CH_IN * UPSAMPLE_KERNEL * WEIGHT_MAG = 16 * 7 * 0.01 = 1.12.
    // IBP may over-approximate but should remain tight.
    assert!(
        width < 20.0,
        "upsample conv with small weights should have tight bounds, got width={width}"
    );

    assert!(
        width > 0.0,
        "upsample conv bounds should have non-zero width, got {width}"
    );

    eprintln!(
        "Generator upsample conv ({CH_IN}->{CH_MID}): bounds=[{lo_min:.6}, {hi_max:.6}], \
         width={width:.4}"
    );
}

// ===========================================================================
// Test 2: Snake activation bounds propagation
// ===========================================================================

/// Snake activation produces bounded output given bounded input.
///
/// Snake(x, alpha) = x + (1/alpha) * sin(alpha*x)^2
///
/// The sin^2 term is bounded in [0, 1], so the additive correction is
/// bounded in [0, 1/alpha]. For alpha >= 1 and input in [-R, R]:
///   output in [-R, R + 1/alpha] ⊂ [-R, R + 1]
///
/// This is the core activation in the Kokoro generator's ResBlocks.
#[test]
fn test_generator_snake_activation_bounds() {
    let def = build_snake_activation(CH_MID, T_LEN);
    def.validate().expect("snake activation def validates");

    let bindings = snake_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CH_MID, T_LEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through snake");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "snake bounds must be finite: [{lo_min}, {hi_max}]"
    );

    // Lower bound should be near -1 (the residual x identity path).
    assert!(
        lo_min >= -2.0,
        "snake lower bound {lo_min} should be >= -2.0 for input in [-1, 1]"
    );
    // Upper bound: x + 1/alpha * sin^2 <= 1 + 1 = 2.
    // IBP over-approximation may widen this somewhat.
    assert!(
        hi_max <= 5.0,
        "snake upper bound {hi_max} should be bounded for input in [-1, 1]"
    );

    assert!(
        width > 0.0,
        "snake bounds should have non-zero width, got {width}"
    );
    assert!(
        width < VACUOUS_THRESHOLD,
        "snake bounds width {width} exceeds vacuous threshold {VACUOUS_THRESHOLD}"
    );

    eprintln!("Generator snake activation: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 3: ResBlock residual add preserves bounds
// ===========================================================================

/// Residual addition `skip + branch` is bounded by sum of component bounds.
///
/// If skip in [-S, S] and branch in [-B, B], then:
///   skip + branch in [-(S+B), S+B]
///
/// IBP computes this exactly for addition. In the generator, the skip path
/// carries the input directly while the branch path is attenuated by small
/// conv weights, so the residual connection stabilizes output bounds.
#[test]
fn test_generator_residual_add_bounds() {
    let def = build_residual_add(CH_MID, T_LEN);
    def.validate().expect("residual add def validates");

    let bindings = residual_add_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // skip in [-1, 1], branch in [-0.3, 0.3] (attenuated by conv weights)
    let total = CH_MID * T_LEN * 2;
    let n = CH_MID * T_LEN;
    let mut lower = vec![-1.0f32; n]; // skip
    lower.extend(vec![-0.3f32; n]); // branch
    let mut upper = vec![1.0f32; n];
    upper.extend(vec![0.3f32; n]);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper).unwrap(),
    )
    .expect("valid input bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through residual add");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    // Exact interval arithmetic: [-1, 1] + [-0.3, 0.3] = [-1.3, 1.3]
    assert!(
        lo_min >= -1.3 - 1e-4,
        "residual lower {lo_min} should be >= -1.3"
    );
    assert!(
        hi_max <= 1.3 + 1e-4,
        "residual upper {hi_max} should be <= 1.3"
    );

    let expected_width = 2.6;
    assert!(
        (width - expected_width).abs() < 0.01,
        "residual width {width} should be approximately {expected_width}"
    );

    eprintln!("Generator residual add: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 4: Style projection injection bounds
// ===========================================================================

/// Style projection produces bounded output via affine modulation.
///
/// Architecture: Linear(style_dim, 2*C) -> split(gamma, beta) ->
///   hidden * gamma + beta (AdaIN-style).
///
/// With small projection weights, gamma and beta are small, so the
/// modulated output stays close to the hidden input range.
///
/// Key property: style injection does NOT cause bound explosion when
/// projection weights are small.
#[test]
fn test_generator_style_projection_bounds() {
    let def = build_style_projection(CH_MID, T_LEN, STYLE_DIM);
    def.validate().expect("style projection def validates");

    let bindings = style_projection_bindings(CH_MID, STYLE_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // hidden in [-1, 1], style in [-0.5, 0.5]
    let n_hidden = CH_MID * T_LEN;
    let n_style = STYLE_DIM;
    let total = n_hidden + n_style;
    let mut lower = vec![-1.0f32; n_hidden];
    lower.extend(vec![-0.5f32; n_style]);
    let mut upper = vec![1.0f32; n_hidden];
    upper.extend(vec![0.5f32; n_style]);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper).unwrap(),
    )
    .expect("valid input bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through style projection");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "style projection bounds must be finite: [{lo_min}, {hi_max}]"
    );

    // With small projection weights (0.01), gamma and beta are small.
    // Output = hidden * (1 + small_gamma) + small_beta, so bounds
    // should be close to the hidden input range.
    assert!(
        width < 50.0,
        "style projection with small weights should produce reasonable bounds, \
         got width={width}"
    );

    eprintln!("Generator style projection: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 5: Generator output tanh activation
// ===========================================================================

/// Generator output tanh bounds must be in [-1, 1].
///
/// The final stage of the generator is Conv1d -> tanh, producing audio
/// samples in [-1, 1]. Tanh is a bounded activation:
///   tanh(x) in (-1, 1) for all finite x.
///
/// IBP propagation through tanh should produce bounds within [-1, 1].
#[test]
fn test_generator_output_tanh_bounds() {
    let def = build_tanh_output(CH_OUT, T_LEN);
    def.validate().expect("tanh output def validates");

    let bindings = tanh_output_bindings(CH_OUT);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CH_OUT, T_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through tanh output");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "tanh output bounds must be finite: [{lo_min}, {hi_max}]"
    );

    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, T_LEN],
        "tanh output must produce [1, T] shape (single audio channel)"
    );

    // tanh output must be within [-1, 1]. IBP may be slightly loose
    // due to interval over-approximation, but should not exceed by much.
    assert!(
        lo_min >= -1.0 - 1e-4,
        "tanh lower bound {lo_min} must be >= -1.0"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "tanh upper bound {hi_max} must be <= 1.0"
    );

    assert!(
        width > 0.0,
        "tanh output should have non-zero width, got {width}"
    );

    eprintln!("Generator tanh output: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 6: Multi-stage upsample chain (512->256->128->64)
// ===========================================================================

/// Multi-stage upsample chain maintains bounded outputs.
///
/// The generator has multiple upsample stages, each halving the channel
/// count: 512 -> 256 -> 128 -> 64 (at toy scale: 16 -> 8 -> 4).
/// This test chains multiple Conv1d stages, propagating IBP bounds
/// sequentially. Output bounds from stage k become input for stage k+1.
///
/// Key property: bounds stay finite through the full upsample chain
/// despite channel dimension changes at each stage.
#[test]
fn test_generator_multi_stage_upsample_chain() {
    // Channel cascade: 16 -> 8 -> 4
    let stages: &[(usize, usize)] = &[
        (CH_IN, CH_MID),  // 16 -> 8
        (CH_MID, CH_OUT), // 8 -> 4
    ];

    let mut current_bounds = uniform_bounds(&[CH_IN, T_LEN], 1.0);

    for (i, &(c_in, c_out)) in stages.iter().enumerate() {
        let def = build_upsample_conv(c_in, c_out, T_LEN);
        def.validate()
            .unwrap_or_else(|e| panic!("upsample stage {i} ({c_in}->{c_out}) def: {e}"));

        let bindings = upsample_conv_bindings(c_in, c_out);
        let graph = tensor_kernel_to_graph(&def, &bindings)
            .unwrap_or_else(|e| panic!("upsample stage {i} graph: {e}"));

        let output = graph
            .propagate_ibp(&current_bounds)
            .unwrap_or_else(|e| panic!("upsample stage {i} IBP: {e}"));
        assert_bounds_valid(&output);

        let (lo, hi) = bounds_min_max(&output);
        let width = hi - lo;
        eprintln!(
            "  Upsample stage {i} ({c_in}->{c_out}): bounds=[{lo:.4}, {hi:.4}], \
             width={width:.4}"
        );

        assert!(
            lo.is_finite() && hi.is_finite(),
            "upsample stage {i} bounds must be finite: [{lo}, {hi}]"
        );
        assert!(
            width < VACUOUS_THRESHOLD,
            "upsample stage {i} width {width} exceeds vacuous threshold {VACUOUS_THRESHOLD}"
        );

        current_bounds = output;
    }

    // Final output after full chain must be finite and non-vacuous.
    let (final_lo, final_hi) = bounds_min_max(&current_bounds);
    let final_width = final_hi - final_lo;
    assert!(
        final_lo.is_finite() && final_hi.is_finite(),
        "multi-stage upsample final bounds must be finite: [{final_lo}, {final_hi}]"
    );
    assert!(
        final_width > 0.0,
        "multi-stage upsample should have non-zero width, got {final_width}"
    );

    // Verify output has the final channel count.
    assert_eq!(
        current_bounds.lower_upper().0.shape(),
        &[CH_OUT, T_LEN],
        "final upsample output shape must be [CH_OUT, T]"
    );

    eprintln!(
        "Generator multi-stage upsample ({CH_IN}->{CH_MID}->{CH_OUT}): \
         final bounds=[{final_lo:.4}, {final_hi:.4}], width={final_width:.4}"
    );
}

// ===========================================================================
// Test 7: Dilated convolution bounds (dilation > 1)
// ===========================================================================

/// Dilated Conv1d with various dilation factors preserves finite bounds.
///
/// The generator's ResBlocks use dilated convolutions (dilation [1, 3, 5])
/// to capture multi-scale temporal patterns. Larger dilation factors increase
/// the effective receptive field without adding parameters.
///
/// IBP propagation through dilated Conv1d should produce bounded outputs
/// regardless of dilation factor.
#[test]
fn test_generator_dilated_conv_bounds() {
    let dilations = [1, 3, 5];
    let input = uniform_bounds(&[CH_MID, T_LEN], 1.0);

    let mut prev_width = 0.0f32;
    for &dilation in &dilations {
        let def = build_dilated_conv1d(CH_MID, T_LEN, dilation);
        def.validate()
            .unwrap_or_else(|e| panic!("dilated conv1d dilation={dilation} def: {e}"));

        let bindings = dilated_conv_bindings(CH_MID);
        let graph = tensor_kernel_to_graph(&def, &bindings)
            .unwrap_or_else(|e| panic!("dilated conv1d dilation={dilation} graph: {e}"));

        let output = graph
            .propagate_ibp(&input)
            .unwrap_or_else(|e| panic!("dilated conv1d dilation={dilation} IBP: {e}"));
        assert_bounds_valid(&output);

        let (lo, hi) = bounds_min_max(&output);
        let width = hi - lo;

        assert!(
            lo.is_finite() && hi.is_finite(),
            "dilated conv (d={dilation}) bounds must be finite: [{lo}, {hi}]"
        );
        assert_eq!(
            output.lower_upper().0.shape(),
            &[CH_MID, T_LEN],
            "dilated conv must preserve temporal dimension"
        );

        // Conv1d with small weights should have tight bounds regardless of dilation.
        // Maximum magnitude: CH_MID * KERNEL_SIZE * WEIGHT_MAG = 8 * 3 * 0.01 = 0.24.
        assert!(
            width < 10.0,
            "dilated conv (d={dilation}) with small weights should have tight bounds, \
             got width={width}"
        );

        eprintln!(
            "  Dilated conv (dilation={dilation}): bounds=[{lo:.6}, {hi:.6}], width={width:.4}"
        );

        // Width should be similar across dilations (same weights, same input range).
        if prev_width > 0.0 {
            let ratio = width / prev_width;
            assert!(
                ratio < 2.0 && ratio > 0.5,
                "dilated conv width ratio {ratio:.4} between d={dilation} and previous \
                 is unexpectedly large"
            );
        }
        prev_width = width;
    }
}

// ===========================================================================
// Test 8: Fused ResBlock with style projection bounds
// ===========================================================================

/// Fused ResBlock with style projection maintains bounded outputs.
///
/// Architecture:
///   InstanceNorm -> Snake -> Conv1d(dilated) -> StyleProjection(AdaIN) + residual
///
/// This combines the core generator operations into a single IBP graph:
/// normalization, activation, convolution, style modulation, and residual.
/// With small weights, the bound expansion should be manageable.
///
/// Key property: the residual connection stabilizes bounds even when
/// style injection adds a multiplicative interaction.
#[test]
fn test_generator_fused_resblock_with_style_bounds() {
    let dilation = 1;
    let def = build_fused_resblock_with_style(CH_MID, T_LEN, STYLE_DIM, dilation);
    def.validate()
        .expect("fused resblock with style def validates");

    let bindings = fused_resblock_style_bindings(CH_MID, STYLE_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // hidden in [-1, 1], style in [-0.5, 0.5]
    let n_hidden = CH_MID * T_LEN;
    let n_style = STYLE_DIM;
    let total = n_hidden + n_style;
    let mut lower = vec![-1.0f32; n_hidden];
    lower.extend(vec![-0.5f32; n_style]);
    let mut upper = vec![1.0f32; n_hidden];
    upper.extend(vec![0.5f32; n_style]);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper).unwrap(),
    )
    .expect("valid input bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through fused resblock with style");
    assert_bounds_valid(&output);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CH_MID, T_LEN],
        "fused resblock output shape must be [C, T]"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "fused resblock bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert!(
        width > 0.0,
        "fused resblock bounds should have non-zero width, got {width}"
    );

    // ResBlock with small weights and style projection should not explode bounds.
    // The residual connection (x + branch) stabilizes the output.
    assert_bounds_width(&output, 200.0, "fused_resblock_with_style");

    eprintln!(
        "Generator fused resblock with style (dilation={dilation}): \
         bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}"
    );
}
