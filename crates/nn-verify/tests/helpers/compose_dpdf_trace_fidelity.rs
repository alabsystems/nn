// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for trace-to-graph translation fidelity.
//!
//! Verifies that `tensor_kernel_to_graph` produces correct NY
//! `GraphNetwork` representations from DynTensor operation traces. Each test
//! builds a specific operation pattern using `TensorBlockBuilder`, translates
//! to a graph, then verifies:
//!
//! 1. The graph translates without error.
//! 2. IBP propagation succeeds and produces finite, non-degenerate bounds.
//! 3. Output shapes match expected dimensions.
//! 4. Bounds satisfy domain-specific constraints (e.g., sigmoid in [0, 1]).
//! 5. CROWN tightening works where applicable (IbpValidated mode).
//!
//! ## Tests:
//!
//! 1. **Linear layer trace (IBP)**: MatMul + Add correctly maps to Linear layer.
//! 2. **Linear layer trace (CROWN)**: CROWN tighter than IBP for linear layer.
//! 3. **RMSNorm decomposition (IBP)**: RMSNorm correctly decomposes for bounds.
//! 4. **RMSNorm decomposition (CROWN)**: CROWN through RMSNorm.
//! 5. **Conv2d trace (IBP)**: Conv2d op maps to conv layer in graph.
//! 6. **Conv2d trace (CROWN)**: CROWN through Conv2d.
//! 7. **SiLU activation trace (IBP)**: x * sigmoid(x) decomposition.
//! 8. **GELU activation trace (IBP)**: GELU activation bounds.
//! 9. **Sigmoid activation trace (IBP)**: Sigmoid bounded in [0, 1].
//! 10. **Sigmoid activation trace (CROWN)**: CROWN through sigmoid.
//! 11. **Softmax decomposition (IBP)**: exp -> sum -> div captured.
//! 12. **Softmax decomposition (CROWN)**: CROWN through softmax.
//! 13. **Residual connection (IBP)**: Add of two branches maps correctly.
//! 14. **Residual connection (CROWN)**: CROWN through residual add.
//! 15. **Reshape preservation (IBP)**: Reshape retains element bounds.
//! 16. **Transpose preservation (IBP)**: Transpose retains element bounds.
//! 17. **Reshape + transpose chain (IBP)**: Shape ops compose correctly.
//! 18. **Linear -> activation -> linear pipeline (IBP)**: Multi-op fidelity.
//! 19. **RMSNorm -> linear -> sigmoid pipeline (IBP)**: Norm + activation.
//! 20. **Full trace pipeline (IBP)**: Conv2d -> reshape -> linear -> softmax.
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, DIM=16, FFN_DIM=32, IN_CH=3, OUT_CH=8, SPATIAL=8
//!
//! Part of #4095: Compose tests for trace-to-graph translation fidelity.

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

/// Sequence length for 2D inputs.
const SEQ_LEN: usize = 4;
/// Hidden dimension.
const DIM: usize = 16;
/// FFN intermediate dimension.
const FFN_DIM: usize = 32;
/// Input channels (RGB).
const IN_CH: usize = 3;
/// Output channels for conv.
const OUT_CH: usize = 8;
/// Spatial size for conv inputs.
const SPATIAL: usize = 8;
/// Kernel size for conv.
const KERNEL_SIZE: usize = 3;
/// Conv output spatial size: (8 - 3 + 2*1) / 1 + 1 = 8 (with padding=1).
const CONV_OUT_SPATIAL: usize = 8;
/// Number of classes for softmax output.
const NUM_CLASSES: usize = 8;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Constant zero bias tensor binding.
fn bias(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// RMSNorm epsilon binding.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32))
}

/// RMSNorm weight (all ones) binding.
fn norm_weight_binding(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 1.0f32))
}

/// Build SiLU activation: SiLU(x) = x * sigmoid(x).
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

// ===========================================================================
// 1. Linear layer trace -- MatMul + Add maps to LayerSpec::Linear (IBP)
// ===========================================================================

fn build_linear_trace_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_linear");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let w = b.add_input("weight", &[FFN_DIM, DIM]);
    let bias_node = b.add_input("bias", &[FFN_DIM]);
    let out = b.add_linear(input, w, Some(bias_node), &[SEQ_LEN, FFN_DIM]);
    b.build(out).expect("valid linear trace kernel")
}

fn linear_trace_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        weight(&[FFN_DIM, DIM]),
        bias(&[FFN_DIM]),
    ]
}

#[test]
fn test_trace_fidelity_linear_ibp() {
    let def = build_linear_trace_kernel();
    let bindings = linear_trace_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through linear layer");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, FFN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity linear IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min < hi_max, "bounds must be non-degenerate");
}

// ===========================================================================
// 2. Linear layer trace (CROWN)
// ===========================================================================

#[test]
fn test_trace_fidelity_linear_crown() {
    let def = build_linear_trace_kernel();
    let bindings = linear_trace_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity linear CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. RMSNorm decomposition (IBP)
// ===========================================================================

fn build_rmsnorm_trace_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_rmsnorm");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let eps = b.add_input("eps", &[1]);
    let rms_w = b.add_input("rms_weight", &[DIM]);
    let out = b.add_rms_norm(input, eps, 1, rms_w, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid RMSNorm trace kernel")
}

fn rmsnorm_trace_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        eps_binding(),
        norm_weight_binding(DIM),
    ]
}

#[test]
fn test_trace_fidelity_rmsnorm_ibp() {
    let def = build_rmsnorm_trace_kernel();
    let bindings = rmsnorm_trace_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through RMSNorm");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity RMSNorm IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min < hi_max, "RMSNorm bounds must be non-degenerate");
}

// ===========================================================================
// 4. RMSNorm decomposition (CROWN)
// ===========================================================================

#[test]
fn test_trace_fidelity_rmsnorm_crown() {
    let def = build_rmsnorm_trace_kernel();
    let bindings = rmsnorm_trace_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity RMSNorm CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. Conv2d trace (IBP)
// ===========================================================================

fn build_conv2d_trace_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_conv2d");
    let input = b.add_input("x", &[IN_CH, SPATIAL, SPATIAL]);
    let w = b.add_input("weight", &[OUT_CH, IN_CH, KERNEL_SIZE, KERNEL_SIZE]);
    let bias_node = b.add_input("bias", &[OUT_CH]);
    let out = b.add_conv2d(
        input,
        w,
        Some(bias_node),
        1, // stride_h
        1, // stride_w
        1, // padding_h
        1, // padding_w
        &[OUT_CH, CONV_OUT_SPATIAL, CONV_OUT_SPATIAL],
    );
    b.build(out).expect("valid Conv2d trace kernel")
}

fn conv2d_trace_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        weight(&[OUT_CH, IN_CH, KERNEL_SIZE, KERNEL_SIZE]),
        bias(&[OUT_CH]),
    ]
}

#[test]
fn test_trace_fidelity_conv2d_ibp() {
    let def = build_conv2d_trace_kernel();
    let bindings = conv2d_trace_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through Conv2d");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[OUT_CH, CONV_OUT_SPATIAL, CONV_OUT_SPATIAL]
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity Conv2d IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min < hi_max, "Conv2d bounds must be non-degenerate");
}

// ===========================================================================
// 6. Conv2d trace (CROWN)
// ===========================================================================

#[test]
fn test_trace_fidelity_conv2d_crown() {
    let def = build_conv2d_trace_kernel();
    let bindings = conv2d_trace_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, SPATIAL, SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity Conv2d CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. SiLU activation trace (IBP): x * sigmoid(x)
// ===========================================================================

fn build_silu_trace_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_silu");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = add_silu(&mut b, input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid SiLU trace kernel")
}

#[test]
fn test_trace_fidelity_silu_ibp() {
    let def = build_silu_trace_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through SiLU");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity SiLU IBP: bounds=[{lo_min}, {hi_max}]");
    // SiLU(-2) ~ -0.238, SiLU(2) ~ 1.762
    assert!(
        lo_min < 0.0,
        "SiLU should have negative lower bound for input [-2, 2]"
    );
    assert!(
        hi_max > 0.0,
        "SiLU should have positive upper bound for input [-2, 2]"
    );
}

// ===========================================================================
// 8. GELU activation trace (IBP)
// ===========================================================================

fn build_gelu_trace_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_gelu");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_gelu(input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid GELU trace kernel")
}

#[test]
fn test_trace_fidelity_gelu_ibp() {
    let def = build_gelu_trace_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through GELU");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity GELU IBP: bounds=[{lo_min}, {hi_max}]");
    // GELU(-2) ~ -0.045, GELU(2) ~ 1.955
    assert!(
        lo_min < 0.0,
        "GELU should have negative lower bound for input [-2, 2]"
    );
    assert!(
        hi_max > 0.0,
        "GELU should have positive upper bound for input [-2, 2]"
    );
}

// ===========================================================================
// 9. Sigmoid activation trace (IBP) -- bounded in [0, 1]
// ===========================================================================

fn build_sigmoid_trace_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_sigmoid");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_sigmoid(input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid sigmoid trace kernel")
}

#[test]
fn test_trace_fidelity_sigmoid_ibp() {
    let def = build_sigmoid_trace_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 3.0);

    let output = graph.propagate_ibp(&input).expect("IBP through sigmoid");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity sigmoid IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. Sigmoid activation trace (CROWN)
// ===========================================================================

#[test]
fn test_trace_fidelity_sigmoid_crown() {
    let def = build_sigmoid_trace_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 3.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity sigmoid CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 11. Softmax decomposition (IBP) -- exp -> sum -> div captured
// ===========================================================================

fn build_softmax_trace_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_softmax");
    let input = b.add_input("x", &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_softmax(input, -1, &[SEQ_LEN, NUM_CLASSES]);
    b.build(out).expect("valid softmax trace kernel")
}

#[test]
fn test_trace_fidelity_softmax_ibp() {
    let def = build_softmax_trace_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, NUM_CLASSES], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through softmax");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity softmax IBP: bounds=[{lo_min}, {hi_max}]");
    // Softmax output must be in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 12. Softmax decomposition (CROWN)
// ===========================================================================

#[test]
fn test_trace_fidelity_softmax_crown() {
    let def = build_softmax_trace_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, NUM_CLASSES], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity softmax CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 13. Residual connection (IBP) -- Add of two branches
// ===========================================================================

fn build_residual_trace_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_residual");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    // Branch: Linear projection (simulating a sublayer)
    let w = b.add_input("proj_weight", &[DIM, DIM]);
    let proj = b.add_linear(input, w, None, &[SEQ_LEN, DIM]);
    let proj_act = b.add_relu(proj, &[SEQ_LEN, DIM]);
    // Residual: input + sublayer(input)
    let out = b.add_binary_add(input, proj_act, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid residual trace kernel")
}

fn residual_trace_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        weight(&[DIM, DIM]),
    ]
}

#[test]
fn test_trace_fidelity_residual_ibp() {
    let def = build_residual_trace_kernel();
    let bindings = residual_trace_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through residual connection");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity residual IBP: bounds=[{lo_min}, {hi_max}]");
    // Residual adds the original input, so output range should be wider than
    // a zero-mean relu-capped branch alone.
    assert!(
        lo_min < 0.0,
        "residual should allow negative values (from skip)"
    );
    assert!(hi_max > 0.0, "residual should allow positive values");
}

// ===========================================================================
// 14. Residual connection (CROWN)
// ===========================================================================

#[test]
fn test_trace_fidelity_residual_crown() {
    let def = build_residual_trace_kernel();
    let bindings = residual_trace_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity residual CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 15. Reshape preservation (IBP) -- shape ops retain element bounds
// ===========================================================================

fn build_reshape_trace_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_reshape");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    // Reshape [4, 16] -> [2, 32] -- same total elements
    let reshaped = b.add_reshape(input, &[SEQ_LEN / 2, DIM * 2]);
    // Wrap in identity layer (AddConstant(0.0)) per engineering rule:
    // graph output NETWORK_INPUT must be wrapped in identity layer.
    let w = b.add_input("id_weight", &[DIM * 2, DIM * 2]);
    let out = b.add_linear(reshaped, w, None, &[SEQ_LEN / 2, DIM * 2]);
    b.build(out).expect("valid reshape trace kernel")
}

fn reshape_trace_bindings() -> Vec<TensorParamBinding> {
    // Use identity-like weight (small magnitude, symmetric)
    vec![
        TensorParamBinding::Variable, // x
        weight(&[DIM * 2, DIM * 2]),
    ]
}

#[test]
fn test_trace_fidelity_reshape_ibp() {
    let def = build_reshape_trace_kernel();
    let bindings = reshape_trace_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through reshape");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN / 2, DIM * 2]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity reshape IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 16. Transpose preservation (IBP)
// ===========================================================================

fn build_transpose_trace_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_transpose");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    // Transpose [4, 16] -> [16, 4]
    let transposed = b.add_transpose(input, &[1, 0], &[DIM, SEQ_LEN]);
    // Follow with linear to avoid bare input at output
    let w = b.add_input("proj_weight", &[FFN_DIM, SEQ_LEN]);
    let out = b.add_linear(transposed, w, None, &[DIM, FFN_DIM]);
    b.build(out).expect("valid transpose trace kernel")
}

fn transpose_trace_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        weight(&[FFN_DIM, SEQ_LEN]),
    ]
}

#[test]
fn test_trace_fidelity_transpose_ibp() {
    let def = build_transpose_trace_kernel();
    let bindings = transpose_trace_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through transpose");

    assert_eq!(output.lower_upper().0.shape(), &[DIM, FFN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity transpose IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 17. Reshape + transpose chain (IBP)
// ===========================================================================

fn build_reshape_transpose_chain_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_reshape_transpose_chain");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    // Reshape [4, 16] -> [4, 4, 4]
    let reshaped = b.add_reshape(input, &[SEQ_LEN, 4, 4]);
    // Transpose [4, 4, 4] -> [4, 4, 4] (swap last two dims)
    let transposed = b.add_transpose(reshaped, &[0, 2, 1], &[SEQ_LEN, 4, 4]);
    // Reshape back to [4, 16]
    let flat = b.add_reshape(transposed, &[SEQ_LEN, DIM]);
    // Linear to produce meaningful output
    let w = b.add_input("proj_weight", &[FFN_DIM, DIM]);
    let out = b.add_linear(flat, w, None, &[SEQ_LEN, FFN_DIM]);
    b.build(out).expect("valid reshape+transpose chain kernel")
}

fn reshape_transpose_chain_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        weight(&[FFN_DIM, DIM]),
    ]
}

#[test]
fn test_trace_fidelity_reshape_transpose_chain_ibp() {
    let def = build_reshape_transpose_chain_kernel();
    let bindings = reshape_transpose_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through reshape+transpose chain");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, FFN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity reshape+transpose chain IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 18. Linear -> activation -> linear pipeline (IBP)
// ===========================================================================

fn build_linear_act_linear_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_linear_act_linear");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    // Linear up-projection
    let w1 = b.add_input("up_weight", &[FFN_DIM, DIM]);
    let up = b.add_linear(input, w1, None, &[SEQ_LEN, FFN_DIM]);
    // GELU activation
    let act = b.add_gelu(up, &[SEQ_LEN, FFN_DIM]);
    // Linear down-projection
    let w2 = b.add_input("down_weight", &[DIM, FFN_DIM]);
    let out = b.add_linear(act, w2, None, &[SEQ_LEN, DIM]);
    b.build(out)
        .expect("valid linear-act-linear pipeline kernel")
}

fn linear_act_linear_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        weight(&[FFN_DIM, DIM]),
        weight(&[DIM, FFN_DIM]),
    ]
}

#[test]
fn test_trace_fidelity_linear_act_linear_ibp() {
    let def = build_linear_act_linear_pipeline_kernel();
    let bindings = linear_act_linear_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through linear-act-linear pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity linear-GELU-linear IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min < hi_max, "pipeline bounds must be non-degenerate");
}

// ===========================================================================
// 19. RMSNorm -> linear -> sigmoid pipeline (IBP)
// ===========================================================================

fn build_rmsnorm_linear_sigmoid_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_rmsnorm_linear_sigmoid");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    // RMSNorm
    let eps = b.add_input("eps", &[1]);
    let rms_w = b.add_input("rms_weight", &[DIM]);
    let normed = b.add_rms_norm(input, eps, 1, rms_w, &[SEQ_LEN, DIM]);
    // Linear projection
    let w = b.add_input("proj_weight", &[NUM_CLASSES, DIM]);
    let b_node = b.add_input("proj_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(normed, w, Some(b_node), &[SEQ_LEN, NUM_CLASSES]);
    // Sigmoid for bounded output
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);
    b.build(out)
        .expect("valid rmsnorm-linear-sigmoid pipeline kernel")
}

fn rmsnorm_linear_sigmoid_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        eps_binding(),
        norm_weight_binding(DIM),
        weight(&[NUM_CLASSES, DIM]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_trace_fidelity_rmsnorm_linear_sigmoid_ibp() {
    let def = build_rmsnorm_linear_sigmoid_kernel();
    let bindings = rmsnorm_linear_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through rmsnorm-linear-sigmoid pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Trace fidelity RMSNorm->linear->sigmoid IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 20. Full trace pipeline: Conv2d -> reshape -> linear -> softmax (IBP)
// ===========================================================================

/// Patch size for the full pipeline (stride = kernel = PATCH_SIZE).
const PATCH_SIZE: usize = 4;
/// Grid size after patching: SPATIAL / PATCH_SIZE.
const GRID_SIZE: usize = SPATIAL / PATCH_SIZE; // 2
/// Total patches: GRID_SIZE^2.
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 4

fn build_full_trace_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("trace_full_pipeline");
    let input = b.add_input("image", &[IN_CH, SPATIAL, SPATIAL]);

    // Conv2d patch embedding: [3, 8, 8] -> [OUT_CH, 2, 2]
    let conv_w = b.add_input("patch_weight", &[OUT_CH, IN_CH, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("patch_bias", &[OUT_CH]);
    let patches = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE, // stride_h
        PATCH_SIZE, // stride_w
        0,          // padding_h
        0,          // padding_w
        &[OUT_CH, GRID_SIZE, GRID_SIZE],
    );

    // Reshape: [OUT_CH, 2, 2] -> [OUT_CH, 4]
    let reshaped = b.add_reshape(patches, &[OUT_CH, NUM_PATCHES]);
    // Transpose: [OUT_CH, 4] -> [4, OUT_CH]
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, OUT_CH]);

    // Linear projection: [4, 8] -> [4, NUM_CLASSES]
    let proj_w = b.add_input("proj_weight", &[NUM_CLASSES, OUT_CH]);
    let proj_b = b.add_input("proj_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(
        transposed,
        proj_w,
        Some(proj_b),
        &[NUM_PATCHES, NUM_CLASSES],
    );

    // Softmax: [4, 8] -> [4, 8]
    let out = b.add_softmax(logits, -1, &[NUM_PATCHES, NUM_CLASSES]);

    b.build(out).expect("valid full trace pipeline kernel")
}

fn full_trace_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        weight(&[OUT_CH, IN_CH, PATCH_SIZE, PATCH_SIZE]),
        bias(&[OUT_CH]),
        weight(&[NUM_CLASSES, OUT_CH]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_trace_fidelity_full_pipeline_ibp() {
    let def = build_full_trace_pipeline_kernel();
    let bindings = full_trace_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Image input: pixels in [0, 1]
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CH, SPATIAL, SPATIAL]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IN_CH, SPATIAL, SPATIAL]), 1.0f32),
    )
    .expect("valid image bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full trace pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_PATCHES, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Trace fidelity full pipeline (Conv2d->reshape->linear->softmax) IBP: \
         bounds=[{lo_min}, {hi_max}]"
    );
    // Softmax output must be in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
    assert!(lo_min < hi_max, "pipeline bounds must be non-degenerate");
}
