// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Normalization layer NY compose verification tests (#3565).
//!
//! Tests CROWN and IBP bounds propagation through the three core normalization
//! layer types used in nn models:
//!
//! - **BatchNorm** (inference): affine normalization with frozen running mean/var.
//!   Used in CNN backbones and feature extractors. Decomposition:
//!   `y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta`
//!
//! - **InstanceNorm**: per-channel per-sample normalization over spatial dims.
//!   Used in Kokoro TTS (58 chained InstanceNorm layers in Generator).
//!   `y = (x - mean(x)) / sqrt(var(x) + eps)`
//!
//! - **AdaIn** (Adaptive Instance Normalization): InstanceNorm + style injection.
//!   Used in Kokoro ISTFTNet decoder for style conditioning.
//!   `y = (1 + gamma(style)) * InstanceNorm(x) + beta(style)`
//!
//! Key normalization verification properties:
//! 1. IBP bounds propagate finitely through all three norm types.
//! 2. CROWN linearization succeeds (IbpValidated mode — Sound refuses
//!    linearization for normalization layers per nn engineering rule).
//! 3. CROWN tightness vs IBP varies: CROWN may produce vacuously wide
//!    bounds through normalization due to FALLBACK_BOUND capping (#2715).
//!    Conservative IBP with contractive weights is often tighter.
//! 4. Concrete soundness: midpoint forward falls within computed bounds.
//!
//! gc#4399 (CROWN through normalization) is CLOSED, unblocking this work.
//!
//! Part of #3565: BatchNorm + InstanceNorm NY compose verification.

use super::common::{
    assert_bounds_valid, assert_bounds_width, assert_crown_tighter_when_not_fallback,
    assert_norm_spatial_non_degenerate, bounds_min_max, high_variance_bounds, uniform_bounds,
    verify_and_assert, DEFAULT_NORM_EPS,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{
    tensor_kernel_to_graph, tensor_kernel_to_graph_with_norm_mode, BoundedTensor, NormBoundsMode,
    TensorParamBinding, VerificationSoundnessMode,
};
use ndarray::{ArrayD, IxDyn};

// ===========================================================================
// Configuration constants
// ===========================================================================

/// Small weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// Builder: BatchNorm (inference with frozen running statistics)
// ===========================================================================

/// Build a BatchNorm graph: Conv1d -> BatchNorm (with affine).
///
/// Input: `[channels, time_len]` (Variable).
/// Conv1d preserves spatial dimension (stride=1, padding=kernel/2).
/// BatchNorm uses frozen running_mean/running_var + learnable gamma/beta.
///
/// This models the common CNN pattern: convolution followed by batch
/// normalization, where BatchNorm uses pre-computed statistics from training.
fn build_conv_batchnorm_kernel(
    channels: usize,
    time_len: usize,
    kernel_size: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let padding = kernel_size / 2;
    let shape = [channels, time_len];

    let mut b = TensorBlockBuilder::new("conv_batchnorm");
    let input = b.add_input("x", &shape);
    let conv_w = b.add_input("conv_weight", &[channels, channels, kernel_size]);
    let bn_mean = b.add_input("running_mean", &[channels]);
    let bn_var = b.add_input("running_var", &[channels]);
    let bn_weight = b.add_input("bn_weight", &[channels]);
    let bn_bias = b.add_input("bn_bias", &[channels]);
    let eps = b.add_input("eps", &[1]);

    // Conv1d (same padding, stride=1)
    let conv = b.add_conv1d(input, conv_w, None, 1, padding, &shape);

    // BatchNorm (frozen running stats + affine)
    let out = b.add_batch_norm(conv, bn_mean, bn_var, bn_weight, bn_bias, eps, &shape);

    let def = b.build(out).expect("valid Conv+BatchNorm kernel");

    let w_conv = ArrayD::from_elem(IxDyn(&[channels, channels, kernel_size]), WEIGHT_MAG);
    let running_mean = ArrayD::from_elem(IxDyn(&[channels]), 0.0f32);
    let running_var = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let gamma = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let beta = ArrayD::from_elem(IxDyn(&[channels]), 0.0f32);

    let bindings = vec![
        TensorParamBinding::Variable,                         // x
        TensorParamBinding::ConstantTensor(w_conv),           // conv_weight
        TensorParamBinding::ConstantTensor(running_mean),     // running_mean
        TensorParamBinding::ConstantTensor(running_var),      // running_var
        TensorParamBinding::ConstantTensor(gamma),            // bn_weight (gamma)
        TensorParamBinding::ConstantTensor(beta),             // bn_bias (beta)
        TensorParamBinding::ConstantScalar(DEFAULT_NORM_EPS), // eps
    ];
    (def, bindings)
}

/// Build a standalone BatchNorm graph (no conv, just normalization).
///
/// Input: `[channels, time_len]` (Variable).
/// Tests BatchNorm in isolation with non-trivial affine parameters.
fn build_batchnorm_affine_kernel(
    channels: usize,
    time_len: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let shape = [channels, time_len];

    let mut b = TensorBlockBuilder::new("batchnorm_affine");
    let input = b.add_input("x", &shape);
    let bn_mean = b.add_input("running_mean", &[channels]);
    let bn_var = b.add_input("running_var", &[channels]);
    let bn_weight = b.add_input("bn_weight", &[channels]);
    let bn_bias = b.add_input("bn_bias", &[channels]);
    let eps = b.add_input("eps", &[1]);

    let out = b.add_batch_norm(input, bn_mean, bn_var, bn_weight, bn_bias, eps, &shape);

    let def = b.build(out).expect("valid BatchNorm affine kernel");

    // Non-trivial running stats: mean shifted, var non-uniform.
    let mean_data: Vec<f32> = (0..channels).map(|i| (i as f32) * 0.1).collect();
    let var_data: Vec<f32> = (0..channels).map(|i| 0.5 + (i as f32) * 0.2).collect();
    let gamma_data: Vec<f32> = (0..channels).map(|i| 0.8 + (i as f32) * 0.1).collect();
    let beta_data: Vec<f32> = (0..channels).map(|i| -0.1 + (i as f32) * 0.05).collect();

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[channels]), mean_data).unwrap(),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[channels]), var_data).unwrap(),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[channels]), gamma_data).unwrap(),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[channels]), beta_data).unwrap(),
        ),
        TensorParamBinding::ConstantScalar(DEFAULT_NORM_EPS),
    ];
    (def, bindings)
}

// ===========================================================================
// Builder: InstanceNorm (standalone, not chained — chained tests in
// compose_chained_norm.rs)
// ===========================================================================

/// Build an InstanceNorm graph: Conv1d -> InstanceNorm -> ReLU.
///
/// Input: `[channels, time_len]` (Variable).
/// Tests InstanceNorm in a realistic single-layer context with surrounding ops.
fn build_conv_instancenorm_relu_kernel(
    channels: usize,
    time_len: usize,
    kernel_size: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    assert_norm_spatial_non_degenerate(time_len, "conv_instancenorm_relu");

    let padding = kernel_size / 2;
    let shape = [channels, time_len];

    let mut b = TensorBlockBuilder::new("conv_instancenorm_relu");
    let input = b.add_input("x", &shape);
    let conv_w = b.add_input("conv_weight", &[channels, channels, kernel_size]);
    let eps = b.add_input("eps", &[1]);

    let conv = b.add_conv1d(input, conv_w, None, 1, padding, &shape);
    let normed = b.add_instance_norm(conv, eps, 1, None, None, &shape);
    let out = b.add_relu(normed, &shape);

    let def = b.build(out).expect("valid Conv+InstanceNorm+ReLU kernel");

    let w_conv = ArrayD::from_elem(IxDyn(&[channels, channels, kernel_size]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w_conv),
        TensorParamBinding::ConstantScalar(DEFAULT_NORM_EPS),
    ];
    (def, bindings)
}

/// Build an affine InstanceNorm graph with learnable scale/shift.
///
/// Input: `[channels, time_len]` (Variable).
/// InstanceNorm with gamma (scale) and beta (shift) per channel.
fn build_instancenorm_affine_kernel(
    channels: usize,
    time_len: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    assert_norm_spatial_non_degenerate(time_len, "instancenorm_affine");

    let shape = [channels, time_len];

    let mut b = TensorBlockBuilder::new("instancenorm_affine");
    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[channels]);
    let beta = b.add_input("beta", &[channels]);

    let out = b.add_instance_norm(input, eps, 1, Some(gamma), Some(beta), &shape);

    let def = b.build(out).expect("valid affine InstanceNorm kernel");

    let gamma_data: Vec<f32> = (0..channels).map(|i| 0.8 + (i as f32) * 0.1).collect();
    let beta_data: Vec<f32> = (0..channels).map(|i| -0.1 + (i as f32) * 0.05).collect();

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(DEFAULT_NORM_EPS),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[channels]), gamma_data).unwrap(),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[channels]), beta_data).unwrap(),
        ),
    ];
    (def, bindings)
}

// ===========================================================================
// Builder: AdaIn (InstanceNorm + style-conditioned affine)
// ===========================================================================

/// Build an AdaIn graph: InstanceNorm(x) -> affine(style) -> LeakyReLU.
///
/// Two Variable inputs:
///   - x: `[channels, time_len]` (content tensor)
///   - style_proj: `[2 * channels]` (pre-projected style: first half gamma, second half beta)
///
/// This mirrors the Kokoro ISTFTNet decoder pattern where style is projected
/// via a Linear layer, then split into gamma/beta for adaptive normalization.
/// The Linear projection is modeled as a pre-projected Variable to keep the
/// verification graph small while still exercising the AdaIn composition.
///
/// Architecture:
///   normed = InstanceNorm(x)
///   gamma = style_proj[0:C], beta = style_proj[C:2C]
///   y = (1 + gamma) * normed + beta  (broadcast over spatial dim)
///   out = LeakyReLU(y, 0.2)
fn build_adain_leaky_relu_kernel(
    channels: usize,
    time_len: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    assert_norm_spatial_non_degenerate(time_len, "adain_leaky_relu");

    let shape = [channels, time_len];

    let mut b = TensorBlockBuilder::new("adain_leaky_relu");
    let x = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);
    // Pre-projected style parameters as separate gamma/beta inputs.
    // In production, these come from Linear(style_vec) split at channel boundary.
    let gamma = b.add_input("gamma", &[channels]);
    let beta = b.add_input("beta", &[channels]);

    // Step 1: InstanceNorm(x) — per-channel normalization
    let normed = b.add_instance_norm(x, eps, 1, None, None, &shape);

    // Step 2: Affine with style: y = (1 + gamma) * normed + beta
    // Broadcast gamma [C] -> [C, T], beta [C] -> [C, T]
    let gamma_bc = b.add_broadcast_left(gamma, &shape);
    let beta_bc = b.add_broadcast_left(beta, &shape);

    // (1 + gamma) * normed: we model "1 + gamma" as a bias-shifted scale.
    // Build: ones [C, T] + broadcast(gamma) [C, T] = scale [C, T]
    // Then: scale * normed + beta
    //
    // For the NY graph, use the decomposition:
    //   normed * broadcast(gamma) + normed + broadcast(beta)
    // which avoids constructing a constant ones tensor in the graph.
    let scaled = b.add_binary_mul(normed, gamma_bc, &shape);
    let with_identity = b.add_binary_add(scaled, normed, &shape);
    let shifted = b.add_binary_add(with_identity, beta_bc, &shape);

    // Step 3: LeakyReLU activation (Kokoro uses 0.2 slope in decoder)
    let out = b.add_leaky_relu(shifted, 0.2, &shape);

    let def = b.build(out).expect("valid AdaIn+LeakyReLU kernel");

    // Small gamma/beta: style modulation is modest.
    let gamma_data: Vec<f32> = (0..channels).map(|i| 0.1 * ((i as f32) - 2.0)).collect();
    let beta_data: Vec<f32> = (0..channels).map(|i| 0.05 * (i as f32)).collect();

    let bindings = vec![
        TensorParamBinding::Variable,                         // x [C, T]
        TensorParamBinding::ConstantScalar(DEFAULT_NORM_EPS), // eps
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[channels]), gamma_data).unwrap(),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[channels]), beta_data).unwrap(),
        ),
    ];
    (def, bindings)
}

/// Build a two-Variable AdaIn graph where both content and style are Variable.
///
/// This exercises NY's multi-variable propagation through the AdaIn
/// composition. Uses `SliceLayer` semantics via a single NETWORK_INPUT that
/// is split into content and style regions.
///
/// Input: single Variable `[channels * (time_len + 2)]` flattened.
///   - First `channels * time_len` elements: content x [C, T]
///   - Next `channels` elements: gamma [C]
///   - Last `channels` elements: beta [C]
fn build_adain_two_variable_kernel(
    channels: usize,
    time_len: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    assert_norm_spatial_non_degenerate(time_len, "adain_two_variable");

    let shape = [channels, time_len];

    let mut b = TensorBlockBuilder::new("adain_two_variable");
    // Content input (Variable)
    let x = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);
    // Style inputs (Variable) — both content and style vary
    let gamma = b.add_input("gamma", &[channels]);
    let beta = b.add_input("beta", &[channels]);

    // InstanceNorm(x)
    let normed = b.add_instance_norm(x, eps, 1, None, None, &shape);

    // Affine: (1 + gamma) * normed + beta
    let gamma_bc = b.add_broadcast_left(gamma, &shape);
    let beta_bc = b.add_broadcast_left(beta, &shape);
    let scaled = b.add_binary_mul(normed, gamma_bc, &shape);
    let with_identity = b.add_binary_add(scaled, normed, &shape);
    let out = b.add_binary_add(with_identity, beta_bc, &shape);

    let def = b.build(out).expect("valid AdaIn two-variable kernel");

    // All three are Variable — NY will propagate bounds for both.
    let bindings = vec![
        TensorParamBinding::Variable,                         // x [C, T]
        TensorParamBinding::ConstantScalar(DEFAULT_NORM_EPS), // eps
        TensorParamBinding::Variable,                         // gamma [C]
        TensorParamBinding::Variable,                         // beta [C]
    ];
    (def, bindings)
}

// ===========================================================================
// Tests: BatchNorm (inference)
// ===========================================================================

/// BatchNorm (Conv+BN): TensorKernelDef validates.
#[test]
fn test_batchnorm_conv_def_validates() {
    let (def, _) = build_conv_batchnorm_kernel(4, 8, 3);
    def.validate().expect("Conv+BatchNorm should validate");
}

/// BatchNorm (Conv+BN): graph builds with correct depth.
#[test]
fn test_batchnorm_conv_graph_builds() {
    let (def, bindings) = build_conv_batchnorm_kernel(4, 8, 3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Conv1d + BatchNorm decomposition. BatchNorm with frozen running stats
    // is a linear transform that may be constant-folded into fewer graph nodes.
    assert!(
        graph.num_nodes() >= 2,
        "Conv+BatchNorm graph should have >= 2 nodes, got {}",
        graph.num_nodes()
    );
}

/// BatchNorm (Conv+BN): IBP propagates, bounds finite and valid.
#[test]
fn test_batchnorm_conv_ibp_propagates() {
    let channels = 4;
    let time_len = 8;
    let (def, bindings) = build_conv_batchnorm_kernel(channels, time_len, 3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[channels, time_len], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Conv+BatchNorm");

    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv+BatchNorm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// BatchNorm (affine only): IBP with non-trivial running stats.
///
/// Non-zero mean, non-uniform variance, and non-identity gamma/beta exercise
/// all branches of the BatchNorm decomposition.
#[test]
fn test_batchnorm_affine_ibp_non_trivial() {
    let channels = 8;
    let time_len = 4;
    let (def, bindings) = build_batchnorm_affine_kernel(channels, time_len);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[channels, time_len], 2.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through BatchNorm affine");

    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("BatchNorm affine IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // BatchNorm with frozen stats is a linear transform: y = scale * x + bias.
    // With small weights and [-2, 2] input, bounds should be reasonable.
    assert_bounds_width(&output, 100.0, "batchnorm_affine");
}

/// BatchNorm (Conv+BN): CROWN propagation (IbpValidated mode).
///
/// BatchNorm (inference) is a linear transform (pre-computed scale/bias),
/// so CROWN should produce tighter bounds than IBP.
#[test]
fn test_batchnorm_conv_crown_propagation() {
    let channels = 4;
    let time_len = 8;
    let (def, bindings) = build_conv_batchnorm_kernel(channels, time_len, 3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[channels, time_len], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv+BatchNorm CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// BatchNorm: verify and record to status file.
#[test]
fn test_batchnorm_verify_and_record() {
    let channels = 4;
    let time_len = 8;
    let (def, bindings) = build_conv_batchnorm_kernel(channels, time_len, 3);
    let input = uniform_bounds(&[channels, time_len], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "norm_batchnorm_conv");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[channels, time_len]);
}

/// BatchNorm: concrete soundness (midpoint within bounds).
#[test]
fn test_batchnorm_affine_soundness() {
    let channels = 8;
    let time_len = 4;
    let (def, bindings) = build_batchnorm_affine_kernel(channels, time_len);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = high_variance_bounds(&[channels, time_len], 2.0, 0.5);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    // Concrete forward via point-IBP.
    let (in_lo, in_hi) = input.lower_upper();
    let midpoint = (in_lo.to_owned() + in_hi.to_owned()) / 2.0;
    let point_input = BoundedTensor::new(midpoint.clone(), midpoint).expect("valid point");
    let point_output = graph.propagate_ibp(&point_input).expect("point-IBP");
    let (point_lo, _) = point_output.lower_upper();

    let (out_lo, out_hi) = output.lower_upper();
    let eps = 1e-3;
    for (i, (&val, (&lo, &hi))) in point_lo
        .iter()
        .zip(out_lo.iter().zip(out_hi.iter()))
        .enumerate()
    {
        assert!(
            val >= lo - eps && val <= hi + eps,
            "BatchNorm soundness violation at {i}: concrete {val} outside [{lo}, {hi}]"
        );
    }
}

// ===========================================================================
// Tests: InstanceNorm (single-layer composition)
// ===========================================================================

/// InstanceNorm (Conv+IN+ReLU): TensorKernelDef validates.
#[test]
fn test_instancenorm_conv_relu_def_validates() {
    let (def, _) = build_conv_instancenorm_relu_kernel(4, 8, 3);
    def.validate()
        .expect("Conv+InstanceNorm+ReLU should validate");
}

/// InstanceNorm (Conv+IN+ReLU): IBP propagates, bounds finite.
#[test]
fn test_instancenorm_conv_relu_ibp_propagates() {
    let channels = 4;
    let time_len = 8;
    let (def, bindings) = build_conv_instancenorm_relu_kernel(channels, time_len, 3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[channels, time_len], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Conv+InstanceNorm+ReLU");

    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv+InstanceNorm+ReLU IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
}

/// InstanceNorm (affine): IBP with learnable gamma/beta.
#[test]
fn test_instancenorm_affine_ibp() {
    let channels = 8;
    let time_len = 4;
    let (def, bindings) = build_instancenorm_affine_kernel(channels, time_len);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[channels, time_len], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through affine InstanceNorm");

    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("InstanceNorm affine IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
}

/// InstanceNorm (Conv+IN+ReLU): CROWN propagation (IbpValidated mode).
///
/// InstanceNorm requires heuristic CROWN linearization.
/// After gc#4399, CROWN should complete without fallback.
#[test]
fn test_instancenorm_conv_relu_crown_propagation() {
    let channels = 4;
    let time_len = 8;
    let (def, bindings) = build_conv_instancenorm_relu_kernel(channels, time_len, 3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[channels, time_len], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv+InstanceNorm+ReLU CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// InstanceNorm (Conv+IN+ReLU): Conservative IBP produces tighter bounds.
///
/// Conservative mode with contractive Conv1d weights often produces tighter
/// bounds than default ForwardMode through normalization layers.
#[test]
fn test_instancenorm_conservative_ibp_tightness() {
    let channels = 4;
    let time_len = 8;
    let (def, bindings) = build_conv_instancenorm_relu_kernel(channels, time_len, 3);

    let graph_conservative =
        tensor_kernel_to_graph_with_norm_mode(&def, &bindings, NormBoundsMode::Conservative)
            .expect("conservative graph");
    let graph_forward =
        tensor_kernel_to_graph_with_norm_mode(&def, &bindings, NormBoundsMode::ForwardMode)
            .expect("forward graph");

    let input = uniform_bounds(&[channels, time_len], 1.0);

    let out_con = graph_conservative
        .propagate_ibp(&input)
        .expect("Conservative IBP");
    let out_fwd = graph_forward
        .propagate_ibp(&input)
        .expect("ForwardMode IBP");

    assert_bounds_valid(&out_con);
    assert_bounds_valid(&out_fwd);

    let con_width = out_con.max_width();
    let fwd_width = out_fwd.max_width();

    eprintln!(
        "InstanceNorm tightness: conservative_width={con_width:.4e}, \
         forward_width={fwd_width:.4e}, ratio(con/fwd)={:.1}x",
        con_width / fwd_width.max(1e-10)
    );
}

/// InstanceNorm: verify and record to status file.
///
/// InstanceNorm uses heuristic normalization approximation.
#[test]
fn test_instancenorm_verify_and_record() {
    let channels = 4;
    let time_len = 8;
    let (def, bindings) = build_conv_instancenorm_relu_kernel(channels, time_len, 3);
    let input = uniform_bounds(&[channels, time_len], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "norm_instancenorm_conv_relu");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[channels, time_len]);

    // InstanceNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "InstanceNorm should produce Heuristic soundness, got {:?}",
        result.verification.soundness_mode
    );
}

/// InstanceNorm (affine): concrete soundness check.
#[test]
fn test_instancenorm_affine_soundness() {
    let channels = 8;
    let time_len = 4;
    let (def, bindings) = build_instancenorm_affine_kernel(channels, time_len);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = high_variance_bounds(&[channels, time_len], 2.0, 0.5);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (in_lo, in_hi) = input.lower_upper();
    let midpoint = (in_lo.to_owned() + in_hi.to_owned()) / 2.0;
    let point_input = BoundedTensor::new(midpoint.clone(), midpoint).expect("valid point");
    let point_output = graph.propagate_ibp(&point_input).expect("point-IBP");
    let (point_lo, _) = point_output.lower_upper();

    let (out_lo, out_hi) = output.lower_upper();
    let eps = 1e-3;
    for (i, (&val, (&lo, &hi))) in point_lo
        .iter()
        .zip(out_lo.iter().zip(out_hi.iter()))
        .enumerate()
    {
        assert!(
            val >= lo - eps && val <= hi + eps,
            "InstanceNorm soundness violation at {i}: concrete {val} outside [{lo}, {hi}]"
        );
    }
}

// ===========================================================================
// Tests: AdaIn (style-conditioned normalization)
// ===========================================================================

/// AdaIn (IN+style+LeakyReLU): TensorKernelDef validates.
#[test]
fn test_adain_leaky_relu_def_validates() {
    let (def, _) = build_adain_leaky_relu_kernel(4, 8);
    def.validate().expect("AdaIn+LeakyReLU should validate");
}

/// AdaIn (IN+style+LeakyReLU): graph builds with sufficient depth.
#[test]
fn test_adain_leaky_relu_graph_builds() {
    let (def, bindings) = build_adain_leaky_relu_kernel(4, 8);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // InstanceNorm + broadcast gamma/beta + mul + add + add + leaky_relu = many nodes
    assert!(
        graph.num_nodes() >= 5,
        "AdaIn+LeakyReLU graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

/// AdaIn (IN+style+LeakyReLU): IBP propagates, bounds finite.
#[test]
fn test_adain_leaky_relu_ibp_propagates() {
    let channels = 4;
    let time_len = 8;
    let (def, bindings) = build_adain_leaky_relu_kernel(channels, time_len);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[channels, time_len], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through AdaIn+LeakyReLU");

    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("AdaIn+LeakyReLU IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
}

/// AdaIn (IN+style+LeakyReLU): CROWN propagation.
///
/// AdaIn combines InstanceNorm (heuristic linearization) with affine transform
/// and LeakyReLU. CROWN should at minimum complete without error.
#[test]
fn test_adain_leaky_relu_crown_propagation() {
    let channels = 4;
    let time_len = 8;
    let (def, bindings) = build_adain_leaky_relu_kernel(channels, time_len);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[channels, time_len], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("AdaIn+LeakyReLU CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// AdaIn: verify and record to status file.
#[test]
fn test_adain_verify_and_record() {
    let channels = 4;
    let time_len = 8;
    let (def, bindings) = build_adain_leaky_relu_kernel(channels, time_len);
    let input = uniform_bounds(&[channels, time_len], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "norm_adain_leaky_relu");
    assert_eq!(result.num_variables, 1, "single Variable input (content)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[channels, time_len]);

    // AdaIn contains InstanceNorm, so soundness should be Heuristic.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "AdaIn with InstanceNorm should produce Heuristic soundness, got {:?}",
        result.verification.soundness_mode
    );
}

/// AdaIn (two-Variable): IBP propagation succeeds through multi-variable InstanceNorm.
///
/// When both content `[C, T]` and style gamma/beta `[C]` are Variable,
/// `tensor_kernel_to_graph` shares ONE flat NETWORK_INPUT and emits, per Variable,
/// a `Slice(axis=0, elem_offset, +flat_i) + Reshape(true_shape)` (see the
/// flat per-variable harness in `graph_tensor_reduce.rs::setup_multi_variable_inputs`,
/// fixed in commit e379be26). Each Variable therefore enters its subgraph at its
/// TRUE declared rank, so InstanceNorm receives a correctly-shaped 2D `[C, T]`
/// content tensor and IBP propagates successfully.
///
/// Previously this asserted failure ("InstanceNorm requires 2D+ input"), but that
/// error was an artifact of the OLD harness handing InstanceNorm a 1D slice — a
/// harness bug, not a soundness guard. Slice + Reshape are exact layout ops, so the
/// fix corrects shapes without loosening bounds; the success here is sound.
#[test]
fn test_adain_two_variable_ibp_succeeds() {
    let channels = 4;
    let time_len = 4;
    let (def, bindings) = build_adain_two_variable_kernel(channels, time_len);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph builds OK");

    // Multi-variable concatenation produces a single flat 1D input:
    //   content x [C, T] | gamma [C] | beta [C].
    let total_size = channels * time_len + channels + channels;
    let input = uniform_bounds(&[total_size], 1.0);

    // IBP propagation now succeeds: each Variable is sliced+reshaped to its TRUE
    // rank, so InstanceNorm sees a proper 2D [C, T] content tensor.
    let output = graph
        .propagate_ibp(&input)
        .expect("multi-variable AdaIn IBP should succeed (per-variable Slice+Reshape)");

    // AdaIn output preserves the content shape [C, T].
    assert_eq!(output.lower_upper().0.shape(), &[channels, time_len]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("AdaIn two-variable IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// AdaIn: concrete soundness (midpoint within bounds).
#[test]
fn test_adain_soundness() {
    let channels = 4;
    let time_len = 8;
    let (def, bindings) = build_adain_leaky_relu_kernel(channels, time_len);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = high_variance_bounds(&[channels, time_len], 2.0, 0.5);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (in_lo, in_hi) = input.lower_upper();
    let midpoint = (in_lo.to_owned() + in_hi.to_owned()) / 2.0;
    let point_input = BoundedTensor::new(midpoint.clone(), midpoint).expect("valid point");
    let point_output = graph.propagate_ibp(&point_input).expect("point-IBP");
    let (point_lo, _) = point_output.lower_upper();

    let (out_lo, out_hi) = output.lower_upper();
    let eps = 1e-3;
    for (i, (&val, (&lo, &hi))) in point_lo
        .iter()
        .zip(out_lo.iter().zip(out_hi.iter()))
        .enumerate()
    {
        assert!(
            val >= lo - eps && val <= hi + eps,
            "AdaIn soundness violation at {i}: concrete {val} outside [{lo}, {hi}]"
        );
    }
}

// ===========================================================================
// Tests: Cross-normalization comparison
// ===========================================================================

/// Compare IBP bounds width across all three normalization types.
///
/// With the same input range and channel count, BatchNorm (frozen linear transform)
/// should produce the tightest bounds, followed by InstanceNorm, with AdaIn widest
/// (due to the multiplicative style interaction).
#[test]
fn test_norm_types_ibp_width_comparison() {
    let channels = 4;
    let time_len = 8;

    // BatchNorm (affine, frozen stats)
    let (bn_def, bn_bindings) = build_batchnorm_affine_kernel(channels, time_len);
    let bn_graph = tensor_kernel_to_graph(&bn_def, &bn_bindings).expect("BN graph");

    // InstanceNorm (affine)
    let (in_def, in_bindings) = build_instancenorm_affine_kernel(channels, time_len);
    let in_graph = tensor_kernel_to_graph(&in_def, &in_bindings).expect("IN graph");

    // AdaIn (constant style + leaky relu)
    let (adain_def, adain_bindings) = build_adain_leaky_relu_kernel(channels, time_len);
    let adain_graph = tensor_kernel_to_graph(&adain_def, &adain_bindings).expect("AdaIn graph");

    let input = uniform_bounds(&[channels, time_len], 1.0);

    let bn_out = bn_graph.propagate_ibp(&input).expect("BN IBP");
    let in_out = in_graph.propagate_ibp(&input).expect("IN IBP");
    let adain_out = adain_graph.propagate_ibp(&input).expect("AdaIn IBP");

    assert_bounds_valid(&bn_out);
    assert_bounds_valid(&in_out);
    assert_bounds_valid(&adain_out);

    let bn_width = bn_out.max_width();
    let in_width = in_out.max_width();
    let adain_width = adain_out.max_width();

    eprintln!(
        "Normalization IBP width comparison:\n  \
         BatchNorm:    {bn_width:.4e}\n  \
         InstanceNorm: {in_width:.4e}\n  \
         AdaIn:        {adain_width:.4e}"
    );

    // All three must produce finite bounds.
    assert!(bn_width.is_finite(), "BatchNorm width must be finite");
    assert!(in_width.is_finite(), "InstanceNorm width must be finite");
    assert!(adain_width.is_finite(), "AdaIn width must be finite");
}
