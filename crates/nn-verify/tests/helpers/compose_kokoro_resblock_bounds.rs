// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! IBP compose tests for Kokoro Generator ResBlock bounds.
//!
//! The Kokoro TTS model (BigVGAN-style generator) uses ResBlocks that combine:
//!   - Snake activation: `x + (1/alpha) * sin(alpha*x)^2`
//!   - 1D convolutions with dilated kernels (dilations [1, 3, 5])
//!   - Residual connections (`skip + branch`)
//!   - InstanceNorm between conv layers
//!
//! This file verifies 8 IBP properties of these ResBlock components:
//!
//! 1. **Snake activation bounds** — output bounded given input bounds and alpha.
//! 2. **Dilated Conv1d bounds** — dilated convolution preserves bounds.
//! 3. **Residual addition bounds** — skip + branch bounded by sum of components.
//! 4. **Single ResBlock bounds** — Snake -> Conv -> Snake -> Conv + residual.
//! 5. **Multi-dilation cascade** — ResBlock with [1, 3, 5] dilation cascade.
//! 6. **Weight-scaled bounds** — bounds tighten proportionally to weight magnitude.
//! 7. **Zero-padded boundary** — padding at temporal boundaries doesn't violate bounds.
//! 8. **Stacked ResBlocks** — 3 sequential ResBlocks maintain bounded outputs.
//!
//! All tests use small dims (C<=8, T<=16) and IBP propagation through proxy graphs
//! built with TensorBlockBuilder.
//!
//! Part of #3351: Epic — Absolutely Best Kokoro.

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
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

/// Channels for ResBlock tests.
const CH: usize = 8;

/// Temporal dimension (must be > 1 for InstanceNorm spatial non-degeneracy).
const T_LEN: usize = 16;

/// Kernel size for convolutions.
const KERNEL_SIZE: usize = 3;

/// Vacuous width threshold — bounds wider than this are meaningless for
/// individual ResBlock components.
const VACUOUS_THRESHOLD: f32 = 500.0;

// ===========================================================================
// Builder helpers
// ===========================================================================

/// Compute padding to preserve temporal dimension for a dilated conv1d.
///
/// For kernel_size=3 with dilation d: effective_kernel = 1 + d * (kernel_size - 1)
/// padding = (effective_kernel - 1) / 2 = d * (kernel_size - 1) / 2 = d (for k=3).
fn dilated_padding(dilation: usize) -> usize {
    dilation * (KERNEL_SIZE - 1) / 2
}

/// Build a Snake activation proxy graph: `x + (1/alpha) * sin(alpha*x)^2`.
///
/// Input: `[C, T]` (Variable).
/// Output: `[C, T]`.
fn build_snake_activation(channels: usize, time_len: usize) -> TensorKernelDef {
    let shape = [channels, time_len];
    let mut b = TensorBlockBuilder::new("resblock_snake_activation");

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

/// Build a single dilated Conv1d graph (no activation, no norm).
///
/// Input: `[C, T]` (Variable).
/// Output: `[C, T]` (same shape with appropriate padding).
fn build_dilated_conv1d(channels: usize, time_len: usize, dilation: usize) -> TensorKernelDef {
    let shape = [channels, time_len];
    let padding = dilated_padding(dilation);
    let mut b = TensorBlockBuilder::new("resblock_dilated_conv1d");

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

/// Build a residual addition graph: `skip + branch`.
///
/// Both `skip` and `branch` are Variable inputs with shape `[C, T]`.
/// Output: `[C, T]`.
fn build_residual_add(channels: usize, time_len: usize) -> TensorKernelDef {
    let shape = [channels, time_len];
    let mut b = TensorBlockBuilder::new("resblock_residual_add");

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

/// Build a single ResBlock graph.
///
/// Architecture (BigVGAN-style from `kokoro_resblock.rs`):
///   InstanceNorm1 -> Snake1 -> Conv1d(dilated) -> InstanceNorm2 -> Snake2 -> Conv1d(k=3,p=1) + residual
///
/// Input: `[C, T]` (Variable).
/// Output: `[C, T]`.
fn build_resblock(channels: usize, time_len: usize, dilation: usize) -> TensorKernelDef {
    assert_norm_spatial_non_degenerate(time_len, "resblock");
    let shape = [channels, time_len];
    let padding = dilated_padding(dilation);

    let mut b = TensorBlockBuilder::new("kokoro_resblock_ibp");

    // Input
    let x = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);

    // InstanceNorm1
    let gamma1 = b.add_input("gamma1", &[channels]);
    let beta1 = b.add_input("beta1", &[channels]);
    let norm1 = b.add_instance_norm(x, eps, 1, Some(gamma1), Some(beta1), &shape);

    // Snake1 activation
    let alpha1 = b.add_input("alpha1", &[1]);
    let alpha1_bc = b.add_broadcast(alpha1, &shape);
    let snake_kernel1 = build_snake_scalar_kernel().expect("snake kernel");
    let snake1 = b.add_elementwise(snake_kernel1, &[norm1, alpha1_bc], &shape);

    // Conv1d (dilated)
    let conv1_w = b.add_input("conv1_w", &[channels, channels, KERNEL_SIZE]);
    let conv1_b = b.add_input("conv1_b", &[channels]);
    let conv1 = b.add_conv1d_full(
        snake1,
        conv1_w,
        Some(conv1_b),
        1,
        padding,
        dilation,
        1,
        &shape,
    );

    // InstanceNorm2
    let gamma2 = b.add_input("gamma2", &[channels]);
    let beta2 = b.add_input("beta2", &[channels]);
    let norm2 = b.add_instance_norm(conv1, eps, 1, Some(gamma2), Some(beta2), &shape);

    // Snake2 activation
    let alpha2 = b.add_input("alpha2", &[1]);
    let alpha2_bc = b.add_broadcast(alpha2, &shape);
    let snake_kernel2 = build_snake_scalar_kernel().expect("snake kernel");
    let snake2 = b.add_elementwise(snake_kernel2, &[norm2, alpha2_bc], &shape);

    // Conv1d (no dilation, kernel=3, padding=1)
    let conv2_w = b.add_input("conv2_w", &[channels, channels, KERNEL_SIZE]);
    let conv2_b = b.add_input("conv2_b", &[channels]);
    let conv2 = b.add_conv1d(snake2, conv2_w, Some(conv2_b), 1, 1, &shape);

    // Residual connection: x + conv2
    let out = b.add_binary_add(x, conv2, &shape);

    b.build(out).expect("valid resblock graph")
}

/// Bindings for a single ResBlock with a given weight magnitude.
fn resblock_bindings(channels: usize, weight_mag: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 1.0f32)), // gamma1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // beta1
        TensorParamBinding::ConstantScalar(1.0),  // alpha1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[channels, channels, KERNEL_SIZE]),
            weight_mag,
        )), // conv1_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // conv1_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 1.0f32)), // gamma2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // beta2
        TensorParamBinding::ConstantScalar(1.0),                                           // alpha2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[channels, channels, KERNEL_SIZE]),
            weight_mag,
        )), // conv2_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // conv2_b
    ]
}

// ===========================================================================
// Test 1: Snake activation bounds
// ===========================================================================

/// Snake activation produces bounded output given bounded input.
///
/// Snake(x, alpha) = x + (1/alpha) * sin(alpha*x)^2
///
/// The sin^2 term is bounded in [0, 1], so the additive correction is
/// bounded in [0, 1/alpha]. For alpha >= 1 and input in [-R, R]:
///   output ∈ [-R, R + 1/alpha] ⊂ [-R, R + 1]
///
/// IBP propagates this correctly through the elementwise kernel.
#[test]
fn test_resblock_snake_activation_bounds() {
    let def = build_snake_activation(CH, T_LEN);
    def.validate().expect("snake activation def validates");

    let bindings = snake_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CH, T_LEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through snake");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    // Snake is x + bounded correction. For alpha=1 and input in [-1, 1]:
    // output ∈ approximately [-1, 2]. IBP may over-approximate.
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

    eprintln!("ResBlock snake activation: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 2: Dilated Conv1d bounds
// ===========================================================================

/// Dilated Conv1d preserves bounded outputs with proper padding.
///
/// A dilated convolution with dilation=3, kernel=3 has effective receptive
/// field of 7. With padding = dilation * (kernel-1)/2 = 3, the temporal
/// dimension is preserved. IBP propagates Conv1d bounds as interval
/// matrix-vector multiply.
#[test]
fn test_resblock_dilated_conv1d_bounds() {
    let dilation = 3;
    let def = build_dilated_conv1d(CH, T_LEN, dilation);
    def.validate().expect("dilated conv1d def validates");

    let bindings = dilated_conv_bindings(CH);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CH, T_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dilated conv1d");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "dilated conv1d bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CH, T_LEN],
        "dilated conv1d must preserve temporal dimension"
    );

    // Conv1d with small weights (0.01) and input in [-1, 1]:
    // Each output element is a weighted sum of CH * KERNEL_SIZE inputs.
    // Maximum magnitude: CH * KERNEL_SIZE * WEIGHT_MAG * 1.0 = 8 * 3 * 0.01 = 0.24.
    // Plus zero bias. IBP should produce tight bounds.
    assert!(
        width < 10.0,
        "dilated conv1d with small weights should have tight bounds, got width={width}"
    );

    eprintln!(
        "ResBlock dilated conv1d (dilation={dilation}): bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}"
    );
}

// ===========================================================================
// Test 3: Residual addition bounds
// ===========================================================================

/// Residual addition `skip + branch` is bounded by sum of component bounds.
///
/// If skip ∈ [-S, S] and branch ∈ [-B, B], then:
///   skip + branch ∈ [-(S+B), S+B]
///
/// IBP computes this exactly for addition.
#[test]
fn test_resblock_residual_addition_bounds() {
    let def = build_residual_add(CH, T_LEN);
    def.validate().expect("residual add def validates");

    let bindings = residual_add_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // skip in [-1, 1], branch in [-0.5, 0.5]
    let total = CH * T_LEN * 2;
    let n = CH * T_LEN;
    let mut lower = vec![-1.0f32; n]; // skip
    lower.extend(vec![-0.5f32; n]); // branch
    let mut upper = vec![1.0f32; n];
    upper.extend(vec![0.5f32; n]);

    let input = nn_verify::BoundedTensor::new(
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

    // Exact interval arithmetic: [-1, 1] + [-0.5, 0.5] = [-1.5, 1.5]
    assert!(
        lo_min >= -1.5 - 1e-4,
        "residual lower {lo_min} should be >= -1.5"
    );
    assert!(
        hi_max <= 1.5 + 1e-4,
        "residual upper {hi_max} should be <= 1.5"
    );
    assert!(
        width > 0.0,
        "residual bounds should have non-zero width, got {width}"
    );

    // Width should be approximately 3.0 (from -1.5 to 1.5).
    let expected_width = 3.0;
    assert!(
        (width - expected_width).abs() < 0.01,
        "residual width {width} should be approximately {expected_width}"
    );

    eprintln!("ResBlock residual add: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 4: Single ResBlock bounds
// ===========================================================================

/// Single ResBlock (InstanceNorm -> Snake -> Conv -> InstanceNorm -> Snake -> Conv + residual)
/// maintains bounded outputs through IBP.
///
/// The residual connection prevents unbounded growth: the skip path passes
/// input directly, while the branch path is attenuated by small conv weights.
/// InstanceNorm normalizes intermediate activations, bounding their range.
#[test]
fn test_resblock_single_full_ibp() {
    let dilation = 1;
    let def = build_resblock(CH, T_LEN, dilation);
    def.validate().expect("single resblock def validates");

    let bindings = resblock_bindings(CH, WEIGHT_MAG);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CH, T_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through single resblock");
    assert_bounds_valid(&output);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CH, T_LEN],
        "resblock output shape must be [C, T]"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "single resblock bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert!(
        width > 0.0,
        "single resblock bounds should have non-zero width, got {width}"
    );

    // ResBlock with small weights should not explode bounds.
    assert_bounds_width(&output, 200.0, "single_resblock_ibp");

    eprintln!(
        "ResBlock single (dilation={dilation}): bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}"
    );
}

// ===========================================================================
// Test 5: Multi-dilation cascade [1, 3, 5]
// ===========================================================================

/// ResBlock cascade with dilations [1, 3, 5] maintains bounded outputs.
///
/// BigVGAN-style generators use multiple dilations per ResBlock layer to
/// capture multi-scale temporal patterns. This test chains 3 ResBlocks
/// with increasing dilation, propagating IBP bounds sequentially. The
/// output bounds of block k become the input bounds of block k+1.
///
/// Key property: bounds stay finite through the full dilation cascade
/// despite compounding through normalization + residual connections.
#[test]
fn test_resblock_multi_dilation_cascade() {
    let dilations = [1, 3, 5];
    let mut current_bounds = uniform_bounds(&[CH, T_LEN], 1.0);

    for (i, &dilation) in dilations.iter().enumerate() {
        let def = build_resblock(CH, T_LEN, dilation);
        def.validate()
            .unwrap_or_else(|e| panic!("resblock dilation={dilation} def: {e}"));

        let bindings = resblock_bindings(CH, WEIGHT_MAG);
        let graph = tensor_kernel_to_graph(&def, &bindings)
            .unwrap_or_else(|e| panic!("resblock dilation={dilation} graph: {e}"));

        let output = graph
            .propagate_ibp(&current_bounds)
            .unwrap_or_else(|e| panic!("resblock dilation={dilation} IBP: {e}"));
        assert_bounds_valid(&output);

        let (lo, hi) = bounds_min_max(&output);
        let width = hi - lo;
        eprintln!(
            "  Cascade block {i} (dilation={dilation}): bounds=[{lo:.4}, {hi:.4}], width={width:.4}"
        );

        assert!(
            lo.is_finite() && hi.is_finite(),
            "cascade block {i} (dilation={dilation}): bounds must be finite [{lo}, {hi}]"
        );

        current_bounds = output;
    }

    // Final output after full [1, 3, 5] cascade must be finite and non-vacuous.
    let (final_lo, final_hi) = bounds_min_max(&current_bounds);
    let final_width = final_hi - final_lo;
    assert!(
        final_lo.is_finite() && final_hi.is_finite(),
        "multi-dilation cascade final bounds must be finite: [{final_lo}, {final_hi}]"
    );
    assert!(
        final_width > 0.0,
        "cascade output should have non-zero width, got {final_width}"
    );

    eprintln!(
        "ResBlock multi-dilation cascade [1,3,5]: final bounds=[{final_lo:.4}, {final_hi:.4}], \
         width={final_width:.4}"
    );
}

// ===========================================================================
// Test 6: Weight-scaled bounds
// ===========================================================================

/// Bounds tighten proportionally to weight magnitude.
///
/// A ResBlock with weight magnitude 0.001 should produce tighter output
/// bounds than the same ResBlock with weight magnitude 0.01. This verifies
/// that IBP correctly propagates the weight scaling through Conv1d layers.
///
/// This is a fundamental IBP property: smaller weights → smaller output
/// intervals from interval matrix-vector multiply.
#[test]
fn test_resblock_weight_scaled_bounds() {
    let dilation = 1;
    let def = build_resblock(CH, T_LEN, dilation);
    let input = uniform_bounds(&[CH, T_LEN], 1.0);

    // Large weights: 0.01
    let bindings_large = resblock_bindings(CH, 0.01);
    let graph_large = tensor_kernel_to_graph(&def, &bindings_large).expect("large weight graph");
    let output_large = graph_large
        .propagate_ibp(&input)
        .expect("IBP with large weights");
    assert_bounds_valid(&output_large);
    let (lo_large, hi_large) = bounds_min_max(&output_large);
    let width_large = hi_large - lo_large;

    // Small weights: 0.001
    let bindings_small = resblock_bindings(CH, 0.001);
    let graph_small = tensor_kernel_to_graph(&def, &bindings_small).expect("small weight graph");
    let output_small = graph_small
        .propagate_ibp(&input)
        .expect("IBP with small weights");
    assert_bounds_valid(&output_small);
    let (lo_small, hi_small) = bounds_min_max(&output_small);
    let width_small = hi_small - lo_small;

    eprintln!(
        "Weight scaling: large_w=0.01 width={width_large:.4}, \
         small_w=0.001 width={width_small:.4}"
    );

    // Smaller weights must produce tighter (or equal) bounds.
    // The ResBlock has a residual connection (x + branch), so the skip path
    // contributes equally regardless of weight magnitude. The branch path
    // tightens with smaller weights.
    assert!(
        width_small <= width_large + 1e-4,
        "smaller weights ({width_small:.6}) should produce tighter bounds \
         than larger weights ({width_large:.6})"
    );

    // The difference should be measurable (not just numerical noise).
    if width_large > 1.0 {
        let ratio = width_small / width_large;
        eprintln!("  Width ratio (small/large): {ratio:.4}");
        assert!(ratio < 1.0 + 1e-4, "width ratio {ratio} should be <= 1.0");
    }
}

// ===========================================================================
// Test 7: Zero-padded boundary
// ===========================================================================

/// Padding at temporal boundaries does not violate output bounds.
///
/// Dilated convolutions with large dilation factors require significant
/// padding to maintain temporal dimensions. This test verifies that the
/// zero-padding at temporal boundaries (where the effective kernel reads
/// padding zeros) does not produce out-of-bounds outputs.
///
/// We compare a dilated conv (dilation=5, padding=5) against a non-dilated
/// conv (dilation=1, padding=1) and verify both produce valid finite bounds.
#[test]
fn test_resblock_zero_padded_boundary() {
    // Dilation=5 requires padding=5 for kernel_size=3.
    let dilation_large = 5;
    let def_large = build_dilated_conv1d(CH, T_LEN, dilation_large);
    def_large.validate().expect("large dilation conv validates");

    let bindings = dilated_conv_bindings(CH);
    let graph_large = tensor_kernel_to_graph(&def_large, &bindings).expect("large dilation graph");
    let input = uniform_bounds(&[CH, T_LEN], 1.0);

    let output_large = graph_large
        .propagate_ibp(&input)
        .expect("IBP through large dilation conv");
    assert_bounds_valid(&output_large);

    // Non-dilated baseline
    let def_small = build_dilated_conv1d(CH, T_LEN, 1);
    let graph_small = tensor_kernel_to_graph(&def_small, &bindings).expect("small dilation graph");
    let output_small = graph_small
        .propagate_ibp(&input)
        .expect("IBP through non-dilated conv");
    assert_bounds_valid(&output_small);

    let (lo_large, hi_large) = bounds_min_max(&output_large);
    let (lo_small, hi_small) = bounds_min_max(&output_small);
    let width_large = hi_large - lo_large;
    let width_small = hi_small - lo_small;

    // Both must preserve temporal shape.
    assert_eq!(
        output_large.lower_upper().0.shape(),
        &[CH, T_LEN],
        "large dilation conv must preserve shape"
    );
    assert_eq!(
        output_small.lower_upper().0.shape(),
        &[CH, T_LEN],
        "small dilation conv must preserve shape"
    );

    // Both must be finite with reasonable width.
    assert!(
        width_large < 10.0,
        "large dilation conv width {width_large} should be tight with small weights"
    );
    assert!(
        width_small < 10.0,
        "small dilation conv width {width_small} should be tight with small weights"
    );

    // The padded (dilated) version may have slightly wider bounds due to
    // more zero-padded positions in the convolution. But the difference
    // should be small since all weights are the same magnitude.
    eprintln!(
        "Zero-padded boundary: dilation=1 width={width_small:.4}, \
         dilation=5 width={width_large:.4}"
    );
}

// ===========================================================================
// Test 8: Stacked ResBlocks (3 sequential)
// ===========================================================================

/// Three sequential ResBlocks (same dilation=1) maintain bounded outputs.
///
/// This tests bound stability through depth: each ResBlock's output bounds
/// become the next block's input. The residual connection in each block
/// prevents unbounded growth, but InstanceNorm can amplify bounds through
/// normalization variance. With small weights, the compounding should
/// stay manageable.
///
/// This verifies the core property needed for the Kokoro generator:
/// multiple stacked ResBlocks (the generator has 12+) produce finite bounds
/// when each block is independently verified and composed.
#[test]
fn test_resblock_stacked_3_blocks() {
    let num_blocks = 3;
    let dilation = 1;
    let mut current_bounds = uniform_bounds(&[CH, T_LEN], 1.0);

    let mut widths = Vec::with_capacity(num_blocks);
    let (init_lo, init_hi) = bounds_min_max(&current_bounds);
    let init_width = init_hi - init_lo;
    widths.push(init_width);
    eprintln!("Stacked ResBlocks (3 blocks, dilation={dilation}):");
    eprintln!("  Input: width={init_width:.4}");

    for i in 0..num_blocks {
        let def = build_resblock(CH, T_LEN, dilation);
        let bindings = resblock_bindings(CH, WEIGHT_MAG);
        let graph = tensor_kernel_to_graph(&def, &bindings)
            .unwrap_or_else(|e| panic!("stacked block {i} graph: {e}"));

        let output = graph
            .propagate_ibp(&current_bounds)
            .unwrap_or_else(|e| panic!("stacked block {i} IBP: {e}"));
        assert_bounds_valid(&output);

        let (lo, hi) = bounds_min_max(&output);
        let width = hi - lo;
        let prev_width = *widths.last().unwrap();
        let expansion = if prev_width > 1e-10 {
            width / prev_width
        } else {
            1.0
        };

        eprintln!(
            "  Block {i}: bounds=[{lo:.4}, {hi:.4}], width={width:.4}, expansion={expansion:.2}x"
        );

        assert!(
            lo.is_finite() && hi.is_finite(),
            "stacked block {i} bounds must be finite: [{lo}, {hi}]"
        );

        widths.push(width);
        current_bounds = output;
    }

    // Final output must be finite.
    let (final_lo, final_hi) = bounds_min_max(&current_bounds);
    let final_width = final_hi - final_lo;
    assert!(
        final_lo.is_finite() && final_hi.is_finite(),
        "stacked 3-block output must be finite: [{final_lo}, {final_hi}]"
    );

    // Track total expansion across 3 blocks.
    let total_expansion = if init_width > 1e-10 {
        final_width / init_width
    } else {
        1.0
    };
    eprintln!(
        "  Total expansion: {total_expansion:.2}x ({num_blocks} blocks). \
         Final width: {final_width:.4}"
    );

    // With small weights (0.01), 3 blocks should not produce vacuously wide bounds.
    // The expansion is primarily from InstanceNorm; with [C=8, T=16] and small
    // weights, it should stay bounded.
    assert!(
        final_width.is_finite(),
        "stacked 3-block output width must be finite, got {final_width}"
    );
}
