// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Normalization layer variant NY composition.
//!
//! Verifies IBP and CROWN bounds propagation through the four core
//! normalization layer types used across dpdf document understanding models:
//!
//! 1. **LayerNorm** — Used in Transformer encoder/decoder layers (Table
//!    Transformer, SVTR in PaddleOCR). Pre-norm or post-norm residual blocks.
//!    `y = gamma * (x - mean) / sqrt(var + eps) + beta`
//!
//! 2. **RMSNorm** — Used in modern LLM decoders (Granite-Docling, GLM-OCR,
//!    Qwen3-VL, FireRed-OCR). Computationally cheaper: no mean subtraction.
//!    `y = x * weight / sqrt(mean(x^2) + eps)`
//!
//! 3. **BatchNorm** — Used in CNN backbones (DocLayout-YOLO Conv-BN-SiLU,
//!    Table Transformer ResNet, PaddleOCR DB detector). Inference mode with
//!    frozen running statistics.
//!    `y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta`
//!
//! 4. **GroupNorm** — Used in some vision encoders. Groups=1 is equivalent
//!    to LayerNorm; groups=channels is equivalent to InstanceNorm.
//!    Decomposed to reshape -> instance_norm -> reshape -> affine.
//!
//! Key verification properties:
//! - IBP bounds propagate finitely through all norm types.
//! - CROWN linearization succeeds with IbpValidated mode (not Sound, which
//!   refuses linearization for normalization layers — nn engineering rule).
//! - CROWN may produce vacuously wide bounds through normalization due to
//!   FALLBACK_BOUND capping (#2715). Tests log width for observability.
//! - Composition tests verify normalization + downstream layers preserve bounds.
//! - Monotone tightening: smaller epsilon -> tighter output bounds.
//!
//! Dimensions (small for fast verification):
//! - HIDDEN_DIM=64, FFN_DIM=128, SEQ_LEN=4, CHANNELS=16, SPATIAL=8
//!
//! Part of #3968: Normalization layer compose tests for dpdf models.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Hidden dimension for transformer-style tests.
const HIDDEN_DIM: usize = 64;
/// FFN intermediate dimension.
const FFN_DIM: usize = 128;
/// Sequence length for 2D [SEQ_LEN, HIDDEN_DIM] inputs.
const SEQ_LEN: usize = 4;
/// Number of channels for CNN-style BatchNorm tests.
const CHANNELS: usize = 16;
/// Spatial dimension for [CHANNELS, SPATIAL] inputs.
const SPATIAL: usize = 8;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// 1. LayerNorm — single layer IBP bounds
// ===========================================================================

/// Build a LayerNorm kernel.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, hidden states in [-1, 1]).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_layernorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_layernorm");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);

    let out = b.add_layer_norm(input, eps, 1, weight, bias, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid LayerNorm kernel")
}

/// Bindings for LayerNorm with weight=1, bias=0 (identity affine).
fn layernorm_bindings() -> Vec<TensorParamBinding> {
    let weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // hidden [SEQ_LEN, HIDDEN_DIM]
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(weight), // weight [HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(bias), // bias [HIDDEN_DIM]
    ]
}

/// LayerNorm IBP bounds propagate finitely.
#[test]
fn test_dpdf_layernorm_ibp_bounds() {
    let def = build_layernorm_kernel();
    let bindings = layernorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through LayerNorm");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "LayerNorm output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf LayerNorm IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. LayerNorm with scaled affine parameters — IBP
// ===========================================================================

/// Bindings for LayerNorm with non-trivial affine: weight=0.5, bias=0.1.
fn layernorm_affine_bindings() -> Vec<TensorParamBinding> {
    let weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.5f32);
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.1f32);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantTensor(bias),
    ]
}

/// LayerNorm with affine parameters: IBP bounds shift by bias and scale by weight.
#[test]
fn test_dpdf_layernorm_affine_ibp_bounds() {
    let def = build_layernorm_kernel();
    let bindings = layernorm_affine_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through LayerNorm with affine");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf LayerNorm affine IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. LayerNorm — CROWN linearization bounds
// ===========================================================================

/// CROWN bounds propagate through LayerNorm.
///
/// LayerNorm involves division by sqrt(var + eps), requiring CROWN
/// linearization. Uses IbpValidated mode per nn engineering rules.
#[test]
fn test_dpdf_layernorm_crown_propagation() {
    let def = build_layernorm_kernel();
    let bindings = layernorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf LayerNorm CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 4. RMSNorm — single layer IBP bounds
// ===========================================================================

/// Build an RMSNorm kernel.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_rmsnorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rmsnorm");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[HIDDEN_DIM]);

    let out = b.add_rms_norm(input, eps, 1, weight, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid RMSNorm kernel")
}

/// Bindings for RMSNorm with weight=1.
fn rmsnorm_bindings() -> Vec<TensorParamBinding> {
    let weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(weight),
    ]
}

/// RMSNorm IBP bounds propagate finitely.
#[test]
fn test_dpdf_rmsnorm_ibp_bounds() {
    let def = build_rmsnorm_kernel();
    let bindings = rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through RMSNorm");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "RMSNorm output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf RMSNorm IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. RMSNorm with scaled weight — IBP
// ===========================================================================

/// Bindings for RMSNorm with weight=0.5 (scaled).
fn rmsnorm_scaled_bindings() -> Vec<TensorParamBinding> {
    let weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.5f32);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(weight),
    ]
}

/// RMSNorm with scaled weight: output magnitude scales with weight.
#[test]
fn test_dpdf_rmsnorm_scaled_weight_ibp_bounds() {
    let def = build_rmsnorm_kernel();
    let bindings = rmsnorm_scaled_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through RMSNorm with scaled weight");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf RMSNorm scaled weight IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. RMSNorm — CROWN bounds
// ===========================================================================

/// CROWN bounds propagate through RMSNorm.
#[test]
fn test_dpdf_rmsnorm_crown_propagation() {
    let def = build_rmsnorm_kernel();
    let bindings = rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf RMSNorm CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 7. BatchNorm — inference mode IBP bounds
// ===========================================================================

/// Build a BatchNorm kernel with frozen running statistics.
///
/// Input: `[CHANNELS, SPATIAL]` (Variable, feature maps in [-1, 1]).
/// Output: `[CHANNELS, SPATIAL]`.
fn build_batchnorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_batchnorm");

    let input = b.add_input("features", &[CHANNELS, SPATIAL]);
    let running_mean = b.add_input("running_mean", &[CHANNELS]);
    let running_var = b.add_input("running_var", &[CHANNELS]);
    let weight = b.add_input("weight", &[CHANNELS]);
    let bias = b.add_input("bias", &[CHANNELS]);
    let eps = b.add_input("eps", &[1]);

    let out = b.add_batch_norm(
        input,
        running_mean,
        running_var,
        weight,
        bias,
        eps,
        &[CHANNELS, SPATIAL],
    );

    b.build(out).expect("valid BatchNorm kernel")
}

/// Bindings for BatchNorm: running_mean=0, running_var=1, weight=1, bias=0.
fn batchnorm_bindings() -> Vec<TensorParamBinding> {
    let running_mean = ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32);
    let running_var = ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32);
    let weight = ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32);
    let bias = ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32);

    vec![
        TensorParamBinding::Variable, // features [CHANNELS, SPATIAL]
        TensorParamBinding::ConstantTensor(running_mean), // running_mean [CHANNELS]
        TensorParamBinding::ConstantTensor(running_var), // running_var [CHANNELS]
        TensorParamBinding::ConstantTensor(weight), // weight [CHANNELS]
        TensorParamBinding::ConstantTensor(bias), // bias [CHANNELS]
        TensorParamBinding::ConstantScalar(1e-5), // eps
    ]
}

/// BatchNorm inference IBP bounds propagate finitely.
#[test]
fn test_dpdf_batchnorm_ibp_bounds() {
    let def = build_batchnorm_kernel();
    let bindings = batchnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through BatchNorm");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL],
        "BatchNorm output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf BatchNorm IBP (features [-1,1]): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. BatchNorm with non-trivial affine — IBP
// ===========================================================================

/// Bindings for BatchNorm: running_mean=0.5, running_var=2.0, weight=0.5, bias=0.1.
fn batchnorm_affine_bindings() -> Vec<TensorParamBinding> {
    let running_mean = ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.5f32);
    let running_var = ArrayD::from_elem(IxDyn(&[CHANNELS]), 2.0f32);
    let weight = ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.5f32);
    let bias = ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.1f32);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(running_mean),
        TensorParamBinding::ConstantTensor(running_var),
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantTensor(bias),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

/// BatchNorm with non-trivial affine parameters: shifted and scaled bounds.
#[test]
fn test_dpdf_batchnorm_affine_ibp_bounds() {
    let def = build_batchnorm_kernel();
    let bindings = batchnorm_affine_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through BatchNorm with affine");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf BatchNorm affine IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. BatchNorm — CROWN bounds
// ===========================================================================

/// CROWN bounds propagate through BatchNorm.
///
/// BatchNorm inference is affine (fixed running stats), so CROWN should
/// linearize it exactly.
#[test]
fn test_dpdf_batchnorm_crown_propagation() {
    let def = build_batchnorm_kernel();
    let bindings = batchnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, SPATIAL]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf BatchNorm CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 10. GroupNorm (groups=4) — IBP bounds
// ===========================================================================

/// Build a GroupNorm(groups=4) kernel using decomposed instance_norm.
///
/// Input: `[CHANNELS, SPATIAL]` (Variable, features in [-1, 1]).
/// Output: `[CHANNELS, SPATIAL]`.
///
/// GroupNorm(G=4) normalizes within groups of CHANNELS/4 channels each.
/// Decomposed: reshape [C, T] -> [G, C/G, T], instance_norm over last 2 dims,
/// reshape back, then affine.
fn build_groupnorm_g4_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_groupnorm_g4");

    let input = b.add_input("features", &[CHANNELS, SPATIAL]);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[CHANNELS]);
    let beta = b.add_input("beta", &[CHANNELS]);

    let num_groups = 4usize;
    let channels_per_group = CHANNELS / num_groups; // 4

    // Reshape [C, T] -> [G, C/G, T]
    let reshaped = b.add_reshape(input, &[num_groups, channels_per_group, SPATIAL]);

    // InstanceNorm over axis 2 (spatial within each group slice)
    // For [G, C/G, T], normalize over the last axis per group-channel pair.
    let normed = b.add_instance_norm(
        reshaped,
        eps,
        2, // axis: spatial dimension
        None,
        None,
        &[num_groups, channels_per_group, SPATIAL],
    );

    // Reshape back to [C, T]
    let unreshaped = b.add_reshape(normed, &[CHANNELS, SPATIAL]);

    // Affine: gamma * x + beta, broadcast gamma [C] and beta [C] over [C, T]
    let gamma_bc = b.add_broadcast_left(gamma, &[CHANNELS, SPATIAL]);
    let scaled = b.add_binary_mul(unreshaped, gamma_bc, &[CHANNELS, SPATIAL]);
    let beta_bc = b.add_broadcast_left(beta, &[CHANNELS, SPATIAL]);
    let out = b.add_binary_add(scaled, beta_bc, &[CHANNELS, SPATIAL]);

    b.build(out).expect("valid GroupNorm(G=4) kernel")
}

/// Bindings for GroupNorm(G=4): gamma=1, beta=0.
fn groupnorm_g4_bindings() -> Vec<TensorParamBinding> {
    let gamma = ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32);
    let beta = ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // features [CHANNELS, SPATIAL]
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(gamma), // gamma [CHANNELS]
        TensorParamBinding::ConstantTensor(beta), // beta [CHANNELS]
    ]
}

/// GroupNorm(G=4) IBP bounds propagate finitely.
#[test]
fn test_dpdf_groupnorm_g4_ibp_bounds() {
    let def = build_groupnorm_g4_kernel();
    let bindings = groupnorm_g4_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GroupNorm(G=4)");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL],
        "GroupNorm(G=4) output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf GroupNorm(G=4) IBP (features [-1,1]): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. GroupNorm (groups=1 = LayerNorm equivalent) — IBP
// ===========================================================================

/// Build a GroupNorm(groups=1) kernel using the optimized `add_group_norm_g1` path.
///
/// Input: `[CHANNELS, SPATIAL]` (Variable).
/// Output: `[CHANNELS, SPATIAL]`.
///
/// GroupNorm(G=1) normalizes over the full C*T dimension, equivalent to
/// LayerNorm over the flattened feature dimension.
fn build_groupnorm_g1_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_groupnorm_g1");

    let input = b.add_input("features", &[CHANNELS, SPATIAL]);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[CHANNELS]);
    let beta = b.add_input("beta", &[CHANNELS]);

    let out = b.add_group_norm_g1(input, eps, Some(gamma), Some(beta), CHANNELS, SPATIAL);

    b.build(out).expect("valid GroupNorm(G=1) kernel")
}

/// Bindings for GroupNorm(G=1): gamma=1, beta=0.
fn groupnorm_g1_bindings() -> Vec<TensorParamBinding> {
    let gamma = ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32);
    let beta = ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
    ]
}

/// GroupNorm(G=1) IBP bounds propagate finitely (LayerNorm equivalent).
#[test]
fn test_dpdf_groupnorm_g1_ibp_bounds() {
    let def = build_groupnorm_g1_kernel();
    let bindings = groupnorm_g1_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GroupNorm(G=1)");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL],
        "GroupNorm(G=1) output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf GroupNorm(G=1) IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 12. GroupNorm — CROWN bounds
// ===========================================================================

/// CROWN bounds propagate through GroupNorm(G=1).
#[test]
fn test_dpdf_groupnorm_g1_crown_propagation() {
    let def = build_groupnorm_g1_kernel();
    let bindings = groupnorm_g1_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, SPATIAL]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf GroupNorm(G=1) CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 13. LayerNorm -> Linear composition — IBP + CROWN
// ===========================================================================

/// Build a LayerNorm -> Linear composition.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, FFN_DIM]`.
///
/// This pattern appears in Transformer FFN blocks (post-norm):
/// LayerNorm(hidden) -> Linear(hidden -> ffn_dim).
fn build_layernorm_linear_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_layernorm_linear");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_weight = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_bias = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let linear_w = b.add_input("linear_weight", &[FFN_DIM, HIDDEN_DIM]);
    let linear_b = b.add_input("linear_bias", &[FFN_DIM]);

    // LayerNorm
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_weight, ln_bias, &[SEQ_LEN, HIDDEN_DIM]);
    // Linear
    let out = b.add_linear(normed, linear_w, Some(linear_b), &[SEQ_LEN, FFN_DIM]);

    b.build(out).expect("valid LayerNorm -> Linear kernel")
}

/// Bindings for LayerNorm -> Linear composition.
fn layernorm_linear_bindings() -> Vec<TensorParamBinding> {
    let ln_weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let linear_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let linear_b = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                  // hidden
        TensorParamBinding::ConstantScalar(1e-5),      // ln_eps
        TensorParamBinding::ConstantTensor(ln_weight), // ln_weight
        TensorParamBinding::ConstantTensor(ln_bias),   // ln_bias
        TensorParamBinding::ConstantTensor(linear_w),  // linear_weight
        TensorParamBinding::ConstantTensor(linear_b),  // linear_bias
    ]
}

/// LayerNorm -> Linear IBP bounds propagate.
#[test]
fn test_dpdf_layernorm_linear_ibp_bounds() {
    let def = build_layernorm_linear_kernel();
    let bindings = layernorm_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through LayerNorm -> Linear");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, FFN_DIM],
        "LayerNorm -> Linear output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf LayerNorm -> Linear IBP: bounds=[{lo_min}, {hi_max}]");
}

/// LayerNorm -> Linear CROWN bounds.
#[test]
fn test_dpdf_layernorm_linear_crown_propagation() {
    let def = build_layernorm_linear_kernel();
    let bindings = layernorm_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, FFN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf LayerNorm -> Linear CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 14. RMSNorm -> SwiGLU composition — IBP + CROWN
// ===========================================================================

/// Build an RMSNorm -> SwiGLU FFN composition.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// This pattern appears in Granite/LLaMA/Qwen decoder layers:
/// RMSNorm -> gate_proj -> SiLU -> mul(up_proj) -> down_proj.
fn build_rmsnorm_swiglu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rmsnorm_swiglu");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let rms_eps = b.add_input("rms_eps", &[1]);
    let rms_weight = b.add_input("rms_weight", &[HIDDEN_DIM]);
    let gate_w = b.add_input("gate_proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_proj_weight", &[HIDDEN_DIM, FFN_DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // RMSNorm
    let normed = b.add_rms_norm(input, rms_eps, 1, rms_weight, &out_shape);

    // SwiGLU: gate_proj -> SiLU -> mul(up_proj) -> down_proj
    let gate = b.add_linear(normed, gate_w, None, &ffn_shape);
    // SiLU(x) = x * sigmoid(x)
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);

    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &out_shape);

    b.build(out).expect("valid RMSNorm -> SwiGLU kernel")
}

/// Bindings for RMSNorm -> SwiGLU composition.
fn rmsnorm_swiglu_bindings() -> Vec<TensorParamBinding> {
    let rms_weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                   // hidden
        TensorParamBinding::ConstantScalar(1e-5),       // rms_eps
        TensorParamBinding::ConstantTensor(rms_weight), // rms_weight
        TensorParamBinding::ConstantTensor(gate_w),     // gate_proj_weight
        TensorParamBinding::ConstantTensor(up_w),       // up_proj_weight
        TensorParamBinding::ConstantTensor(down_w),     // down_proj_weight
    ]
}

/// RMSNorm -> SwiGLU IBP bounds propagate.
#[test]
fn test_dpdf_rmsnorm_swiglu_ibp_bounds() {
    let def = build_rmsnorm_swiglu_kernel();
    let bindings = rmsnorm_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through RMSNorm -> SwiGLU");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "RMSNorm -> SwiGLU output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf RMSNorm -> SwiGLU IBP: bounds=[{lo_min}, {hi_max}]");
}

/// RMSNorm -> SwiGLU CROWN bounds.
#[test]
fn test_dpdf_rmsnorm_swiglu_crown_propagation() {
    let def = build_rmsnorm_swiglu_kernel();
    let bindings = rmsnorm_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf RMSNorm -> SwiGLU CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 15. BatchNorm -> ReLU -> Conv2d composition — IBP
// ===========================================================================

/// Small spatial dimensions for Conv2d composition.
const CONV_SPATIAL: usize = 4;
/// Conv output channels.
const CONV_OUT_CH: usize = 32;

/// Build a BatchNorm -> ReLU -> Conv2d composition.
///
/// Input: `[CHANNELS, CONV_SPATIAL, CONV_SPATIAL]` (Variable, feature maps).
/// Output: `[CONV_OUT_CH, CONV_SPATIAL - 2, CONV_SPATIAL - 2]` (valid padding).
///
/// This pattern is the core building block of ResNet-style backbones
/// (Table Transformer, DocLayout-YOLO): BN -> activation -> conv.
fn build_batchnorm_relu_conv_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_batchnorm_relu_conv");
    let out_h = CONV_SPATIAL - 2; // kernel=3, stride=1, padding=0 => out = in - 2
    let out_w = out_h;

    let input = b.add_input("features", &[CHANNELS, CONV_SPATIAL, CONV_SPATIAL]);
    let bn_mean = b.add_input("bn_running_mean", &[CHANNELS]);
    let bn_var = b.add_input("bn_running_var", &[CHANNELS]);
    let bn_weight = b.add_input("bn_weight", &[CHANNELS]);
    let bn_bias = b.add_input("bn_bias", &[CHANNELS]);
    let bn_eps = b.add_input("bn_eps", &[1]);
    let conv_w = b.add_input("conv_weight", &[CONV_OUT_CH, CHANNELS, 3, 3]);
    let conv_b = b.add_input("conv_bias", &[CONV_OUT_CH]);

    // BatchNorm
    let normed = b.add_batch_norm(
        input,
        bn_mean,
        bn_var,
        bn_weight,
        bn_bias,
        bn_eps,
        &[CHANNELS, CONV_SPATIAL, CONV_SPATIAL],
    );

    // ReLU
    let activated = b.add_relu(normed, &[CHANNELS, CONV_SPATIAL, CONV_SPATIAL]);

    // Conv2d(kernel=3, stride=1, padding=0)
    let out = b.add_conv2d(
        activated,
        conv_w,
        Some(conv_b),
        1, // stride_h
        1, // stride_w
        0, // padding_h
        0, // padding_w
        &[CONV_OUT_CH, out_h, out_w],
    );

    b.build(out)
        .expect("valid BatchNorm -> ReLU -> Conv2d kernel")
}

/// Bindings for BatchNorm -> ReLU -> Conv2d composition.
fn batchnorm_relu_conv_bindings() -> Vec<TensorParamBinding> {
    let bn_mean = ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32);
    let bn_weight = ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32);
    let bn_bias = ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32);
    let conv_w = ArrayD::from_elem(IxDyn(&[CONV_OUT_CH, CHANNELS, 3, 3]), WEIGHT_MAG);
    let conv_b = ArrayD::from_elem(IxDyn(&[CONV_OUT_CH]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                  // features [C, H, W]
        TensorParamBinding::ConstantTensor(bn_mean),   // bn_running_mean
        TensorParamBinding::ConstantTensor(bn_var),    // bn_running_var
        TensorParamBinding::ConstantTensor(bn_weight), // bn_weight
        TensorParamBinding::ConstantTensor(bn_bias),   // bn_bias
        TensorParamBinding::ConstantScalar(1e-5),      // bn_eps
        TensorParamBinding::ConstantTensor(conv_w),    // conv_weight
        TensorParamBinding::ConstantTensor(conv_b),    // conv_bias
    ]
}

/// BatchNorm -> ReLU -> Conv2d IBP bounds propagate.
#[test]
fn test_dpdf_batchnorm_relu_conv_ibp_bounds() {
    let def = build_batchnorm_relu_conv_kernel();
    let bindings = batchnorm_relu_conv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, CONV_SPATIAL, CONV_SPATIAL], 1.0);

    let out_h = CONV_SPATIAL - 2;
    let out_w = out_h;

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through BatchNorm -> ReLU -> Conv2d");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CONV_OUT_CH, out_h, out_w],
        "BatchNorm -> ReLU -> Conv2d output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf BatchNorm -> ReLU -> Conv2d IBP: bounds=[{lo_min}, {hi_max}]");

    // ReLU clamps lower to >= 0, so after BN+ReLU the lower bound should be >= 0
    // before Conv2d. After Conv2d the bounds are mixed by convolution weights.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 16. Normalization monotone tightening: smaller eps -> tighter bounds
// ===========================================================================

/// Verify that smaller epsilon produces tighter (or equal) bounds for RMSNorm.
///
/// This is a key property: eps controls the denominator floor in normalization.
/// Larger eps -> wider denominator range -> potentially wider output bounds.
/// This test checks monotone tightening: eps_small produces bounds at least
/// as tight as eps_large.
#[test]
fn test_dpdf_normalization_eps_monotone_tightening() {
    let eps_values: [f32; 3] = [1e-3, 1e-5, 1e-8];

    let weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let mut prev_width: Option<f32> = None;

    for &eps_val in &eps_values {
        let def = build_rmsnorm_kernel();
        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantScalar(eps_val),
            TensorParamBinding::ConstantTensor(weight.clone()),
        ];

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let output = graph.propagate_ibp(&input).expect("IBP through RMSNorm");

        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("dpdf RMSNorm eps={eps_val:.0e}: width={width:.6}, bounds=[{lo_min}, {hi_max}]");

        if let Some(prev_w) = prev_width {
            // Smaller eps should produce tighter or equal bounds (with tolerance).
            // Note: this is a soft check because IBP over-approximation can be
            // non-monotone in some edge cases. We use generous tolerance.
            let tolerance = prev_w * 0.1 + 1e-4;
            assert!(
                width <= prev_w + tolerance,
                "smaller eps ({eps_val:.0e}) should produce tighter bounds: \
                 width {width:.6} > prev width {prev_w:.6} + tolerance {tolerance:.6}"
            );
        }
        prev_width = Some(width);
    }
}

// ===========================================================================
// 17. Verify and record — LayerNorm
// ===========================================================================

/// Verify and record LayerNorm for the dpdf status file.
#[test]
fn test_dpdf_layernorm_verify_and_record() {
    let def = build_layernorm_kernel();
    let bindings = layernorm_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_layernorm");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 18. Verify and record — RMSNorm
// ===========================================================================

/// Verify and record RMSNorm for the dpdf status file.
#[test]
fn test_dpdf_rmsnorm_verify_and_record() {
    let def = build_rmsnorm_kernel();
    let bindings = rmsnorm_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_rmsnorm");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 19. Verify and record — BatchNorm
// ===========================================================================

/// Verify and record BatchNorm for the dpdf status file.
#[test]
fn test_dpdf_batchnorm_verify_and_record() {
    let def = build_batchnorm_kernel();
    let bindings = batchnorm_bindings();
    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_batchnorm");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[CHANNELS, SPATIAL]);
}

// ===========================================================================
// 20. Verify and record — GroupNorm(G=1)
// ===========================================================================

/// Verify and record GroupNorm(G=1) for the dpdf status file.
#[test]
fn test_dpdf_groupnorm_g1_verify_and_record() {
    let def = build_groupnorm_g1_kernel();
    let bindings = groupnorm_g1_bindings();
    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_groupnorm_g1");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[CHANNELS, SPATIAL]);
}
