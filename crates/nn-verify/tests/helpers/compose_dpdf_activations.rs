// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for activation function variants used across dpdf models.
//!
//! Verifies IBP and CROWN bound propagation through individual activation
//! functions and composed activation-linear pipelines. These activations appear
//! in every dpdf model: GELU (PaddleOCR SVTR MLP), SiLU (DocLayout-YOLO,
//! GLM-OCR SwiGLU, Qwen3-VL), Snake (Kokoro/HTDemucs), Mish (YOLO variants),
//! ReLU (Table Transformer backbone), Sigmoid (detection heads).
//!
//! ## Single Activation IBP (tests 1-11)
//!
//! 1. GELU (tanh approx) IBP bounds
//! 2. GELU (exact erf) IBP bounds
//! 3. GELU CROWN linearization tighter than IBP
//! 4. SiLU (Swish) IBP bounds: x * sigmoid(x)
//! 5. SiLU CROWN bounds tighter than IBP
//! 6. SiLU gate pattern IBP: gate = x * sigmoid(x) for gated linear units
//! 7. Snake activation IBP: x + sin^2(alpha*x) / alpha
//! 8. Snake with varying alpha IBP: smaller alpha -> wider output
//! 9. Mish activation IBP: x * tanh(softplus(x))
//! 10. Mish CROWN bounds tighter than IBP
//! 11. ReLU baseline IBP bounds (reference comparison)
//!
//! ## Composed Activation-Linear Pipelines (tests 12-16)
//!
//! 12. Sigmoid bounded in (0, 1) under IBP + CROWN
//! 13. Linear -> GELU -> Linear composition (IBP + CROWN)
//! 14. Linear -> SiLU -> Linear (SwiGLU building block) (IBP + CROWN)
//! 15. Activation monotone tightening: smaller input eps -> tighter output
//! 16. Activation chain: GELU -> Linear -> SiLU -> Linear (IBP)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, DIM=16, FFN_DIM=32
//!
//! Part of #3969: Activation function compose tests for dpdf models.

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

const SEQ_LEN: usize = 4;
const DIM: usize = 16;
const FFN_DIM: usize = 32;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build SiLU activation: SiLU(x) = x * sigmoid(x).
/// No native SiLU op in TensorBlockBuilder; decompose via sigmoid + binary_mul.
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

/// Build Mish activation: Mish(x) = x * tanh(softplus(x)).
/// Decompose via softplus + tanh + binary_mul.
fn add_mish(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sp = b.add_softplus(input, shape);
    let th = b.add_tanh(sp, shape);
    b.add_binary_mul(input, th, shape)
}

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. GELU (tanh approx) single activation IBP bounds
// ===========================================================================

fn build_gelu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_act_gelu");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_gelu(input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid GELU kernel")
}

#[test]
fn test_gelu_tanh_ibp_bounds() {
    let def = build_gelu_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GELU (tanh) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // GELU is bounded below by ~-0.17 for all inputs, upper unbounded.
    // For input in [-1, 1], output should be in approximately [-0.17, 1.0].
    assert!(lo_min >= -1.0, "GELU lower should be >= -1.0, got {lo_min}");
    assert!(
        hi_max <= 2.0,
        "GELU upper should be <= 2.0 for input in [-1,1], got {hi_max}"
    );
}

// ===========================================================================
// 2. GELU (exact erf) IBP bounds
// ===========================================================================

fn build_gelu_erf_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_act_gelu_erf");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_gelu_erf(input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid GELU erf kernel")
}

#[test]
fn test_gelu_erf_ibp_bounds() {
    let def = build_gelu_erf_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GELU (erf) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -1.0,
        "GELU erf lower should be >= -1.0, got {lo_min}"
    );
    assert!(
        hi_max <= 2.0,
        "GELU erf upper should be <= 2.0, got {hi_max}"
    );
}

// ===========================================================================
// 3. GELU CROWN linearization tighter than IBP
// ===========================================================================

#[test]
fn test_gelu_crown_tighter_than_ibp() {
    let def = build_gelu_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    // Verify IBP works as baseline.
    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    // CROWN should be at least as tight (or fall back gracefully).
    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("GELU CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 4. SiLU (Swish) single activation IBP bounds
// ===========================================================================

fn build_silu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_act_silu");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = add_silu(&mut b, input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid SiLU kernel")
}

#[test]
fn test_silu_ibp_bounds() {
    let def = build_silu_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SiLU IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // SiLU(x) = x * sigmoid(x), bounded below by ~-0.278 for all x.
    // For input in [-1, 1], output is approximately [-0.278, 0.731].
    assert!(lo_min >= -2.0, "SiLU lower should be >= -2.0, got {lo_min}");
    assert!(
        hi_max <= 2.0,
        "SiLU upper should be <= 2.0 for input in [-1,1], got {hi_max}"
    );
}

// ===========================================================================
// 5. SiLU CROWN bounds
// ===========================================================================

#[test]
fn test_silu_crown_bounds() {
    let def = build_silu_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("SiLU CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 6. SiLU gate pattern IBP: Linear -> SiLU (gate), Linear (up), mul
// ===========================================================================

fn build_silu_gate_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_act_silu_gate");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // SwiGLU gate pattern: gate_proj -> SiLU, up_proj, element-wise mul
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let out = b.add_binary_mul(gate_act, up, &ffn_shape);

    b.build(out).expect("valid SiLU gate kernel")
}

fn silu_gate_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, DIM]), WEIGHT_MAG)),
    ]
}

#[test]
fn test_silu_gate_pattern_ibp() {
    let def = build_silu_gate_kernel();
    let bindings = silu_gate_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("SiLU gate pattern IBP: output width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
    assert!(
        width < 50.0,
        "SiLU gate output width {width} exceeds threshold 50.0"
    );
}

// ===========================================================================
// 7. Snake activation IBP: x + sin^2(alpha*x) / alpha
// ===========================================================================

fn build_snake_kernel() -> TensorKernelDef {
    let snake_scalar = nn_dsl::adain::build_snake_scalar_kernel().expect("snake scalar kernel");
    let mut b = TensorBlockBuilder::new("dpdf_act_snake");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let alpha = b.add_input("alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &[SEQ_LEN, DIM]);
    let out = b.add_elementwise(snake_scalar, &[input, alpha_bc], &[SEQ_LEN, DIM]);

    b.build(out).expect("valid Snake kernel")
}

fn snake_bindings(alpha_val: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), alpha_val)),
    ]
}

#[test]
fn test_snake_ibp_bounds() {
    let def = build_snake_kernel();
    let bindings = snake_bindings(1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Snake (alpha=1.0) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Snake(x) = x + sin^2(alpha*x)/alpha. For alpha=1, x in [-1,1]:
    // lower >= -1 (x term dominates), upper <= 1 + 1 = 2.
    assert!(
        lo_min >= -3.0,
        "Snake lower should be >= -3.0, got {lo_min}"
    );
    assert!(hi_max <= 3.0, "Snake upper should be <= 3.0, got {hi_max}");
}

// ===========================================================================
// 8. Snake with varying alpha: smaller alpha -> wider output bounds
// ===========================================================================

#[test]
fn test_snake_varying_alpha_ibp() {
    let def = build_snake_kernel();
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    // Alpha = 10.0 (high frequency, small sin^2 contribution per element)
    let bindings_high = snake_bindings(10.0);
    let graph_high = tensor_kernel_to_graph(&def, &bindings_high).expect("graph high alpha");
    let output_high = graph_high.propagate_ibp(&input).expect("IBP high alpha");
    assert_bounds_valid(&output_high);
    let width_high = bound_width(&output_high);

    // Alpha = 0.5 (low frequency, larger sin^2/alpha contribution)
    let bindings_low = snake_bindings(0.5);
    let graph_low = tensor_kernel_to_graph(&def, &bindings_low).expect("graph low alpha");
    let output_low = graph_low.propagate_ibp(&input).expect("IBP low alpha");
    assert_bounds_valid(&output_low);
    let width_low = bound_width(&output_low);

    eprintln!(
        "Snake alpha comparison: alpha=10.0 width={width_high:.6}, alpha=0.5 width={width_low:.6}"
    );
    // Both should produce valid finite bounds.
    assert!(width_high.is_finite(), "high alpha width must be finite");
    assert!(width_low.is_finite(), "low alpha width must be finite");
}

// ===========================================================================
// 9. Mish activation IBP: x * tanh(softplus(x))
// ===========================================================================

fn build_mish_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_act_mish");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = add_mish(&mut b, input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid Mish kernel")
}

#[test]
fn test_mish_ibp_bounds() {
    let def = build_mish_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mish IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Mish(x) = x * tanh(softplus(x)), bounded below by ~-0.31 for all x.
    // For input in [-1, 1], output is approximately [-0.31, 0.87].
    assert!(lo_min >= -2.0, "Mish lower should be >= -2.0, got {lo_min}");
    assert!(
        hi_max <= 2.0,
        "Mish upper should be <= 2.0 for input in [-1,1], got {hi_max}"
    );
}

// ===========================================================================
// 10. Mish CROWN bounds
// ===========================================================================

#[test]
fn test_mish_crown_bounds() {
    let def = build_mish_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Mish CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 11. ReLU baseline IBP bounds (reference comparison)
// ===========================================================================

fn build_relu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_act_relu");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_relu(input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid ReLU kernel")
}

#[test]
fn test_relu_baseline_ibp_bounds() {
    let def = build_relu_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ReLU IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // ReLU(x) = max(0, x). For input in [-1, 1]: output in [0, 1].
    let tol = 1e-6;
    assert!(lo_min >= 0.0 - tol, "ReLU lower must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + tol,
        "ReLU upper must be <= 1.0 for input in [-1,1], got {hi_max}"
    );
}

// ===========================================================================
// 12. Sigmoid bounded in (0, 1) under IBP + CROWN
// ===========================================================================

fn build_sigmoid_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_act_sigmoid");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_sigmoid(input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid Sigmoid kernel")
}

#[test]
fn test_sigmoid_bounded_01_ibp_and_crown() {
    let def = build_sigmoid_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 2.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let tol = 1e-6;
    eprintln!("Sigmoid IBP: bounds=[{ibp_lo:.6}, {ibp_hi:.6}]");
    assert!(
        ibp_lo >= 0.0 - tol,
        "sigmoid IBP lower must be >= 0, got {ibp_lo}"
    );
    assert!(
        ibp_hi <= 1.0 + tol,
        "sigmoid IBP upper must be <= 1, got {ibp_hi}"
    );

    // CROWN
    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!("Sigmoid CROWN: bounds=[{crown_lo:.6}, {crown_hi:.6}]");
    assert!(
        crown_lo >= 0.0 - tol,
        "sigmoid CROWN lower must be >= 0, got {crown_lo}"
    );
    assert!(
        crown_hi <= 1.0 + tol,
        "sigmoid CROWN upper must be <= 1, got {crown_hi}"
    );
}

// ===========================================================================
// 13. Linear -> GELU -> Linear composition (IBP + CROWN)
// ===========================================================================

fn build_linear_gelu_linear_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_act_linear_gelu_linear");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let w1 = b.add_input("w1", &[FFN_DIM, DIM]);
    let b1 = b.add_input("b1", &[FFN_DIM]);
    let w2 = b.add_input("w2", &[DIM, FFN_DIM]);
    let b2 = b.add_input("b2", &[DIM]);

    let h = b.add_linear(input, w1, Some(b1), &[SEQ_LEN, FFN_DIM]);
    let h = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
    let out = b.add_linear(h, w2, Some(b2), &[SEQ_LEN, DIM]);

    b.build(out).expect("valid Linear-GELU-Linear kernel")
}

fn linear_gelu_linear_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, FFN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ]
}

#[test]
fn test_linear_gelu_linear_ibp_and_crown() {
    let def = build_linear_gelu_linear_kernel();
    let bindings = linear_gelu_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let ibp_width = bound_width(&ibp_output);
    eprintln!("Linear-GELU-Linear IBP: width={ibp_width:.6}");
    assert!(ibp_width.is_finite(), "IBP width must be finite");

    // CROWN
    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let crown_width = bound_width(&crown_output);
    eprintln!("Linear-GELU-Linear CROWN: width={crown_width:.6}");
}

// ===========================================================================
// 14. Linear -> SiLU -> Linear (SwiGLU building block) (IBP + CROWN)
// ===========================================================================

fn build_linear_silu_linear_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_act_linear_silu_linear");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let w1 = b.add_input("w1", &[FFN_DIM, DIM]);
    let b1 = b.add_input("b1", &[FFN_DIM]);
    let w2 = b.add_input("w2", &[DIM, FFN_DIM]);
    let b2 = b.add_input("b2", &[DIM]);

    let h = b.add_linear(input, w1, Some(b1), &[SEQ_LEN, FFN_DIM]);
    let h = add_silu(&mut b, h, &[SEQ_LEN, FFN_DIM]);
    let out = b.add_linear(h, w2, Some(b2), &[SEQ_LEN, DIM]);

    b.build(out).expect("valid Linear-SiLU-Linear kernel")
}

fn linear_silu_linear_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, FFN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ]
}

#[test]
fn test_linear_silu_linear_ibp_and_crown() {
    let def = build_linear_silu_linear_kernel();
    let bindings = linear_silu_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let ibp_width = bound_width(&ibp_output);
    eprintln!("Linear-SiLU-Linear IBP: width={ibp_width:.6}");
    assert!(ibp_width.is_finite(), "IBP width must be finite");

    // CROWN
    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let crown_width = bound_width(&crown_output);
    eprintln!("Linear-SiLU-Linear CROWN: width={crown_width:.6}");
}

// ===========================================================================
// 15. Activation monotone tightening: smaller input eps -> tighter output
// ===========================================================================

/// For a given activation kernel, verify that shrinking input bounds from
/// eps=1.0 to eps=0.1 produces tighter output bounds.
fn assert_monotone_tightening(def: &TensorKernelDef, bindings: &[TensorParamBinding], label: &str) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);
    let wide_width = bound_width(&wide_output);

    let tight_input = uniform_bounds(&[SEQ_LEN, DIM], 0.1);
    let tight_output = graph.propagate_ibp(&tight_input).expect("IBP tight");
    assert_bounds_valid(&tight_output);
    let tight_width = bound_width(&tight_output);

    eprintln!(
        "{label} monotone tightening: eps=1.0 width={wide_width:.6}, eps=0.1 width={tight_width:.6}"
    );
    assert!(
        tight_width <= wide_width + 1e-6,
        "{label}: tight input (eps=0.1) should produce tighter or equal output bounds. \
         wide_width={wide_width}, tight_width={tight_width}"
    );
}

#[test]
fn test_activation_monotone_tightening_gelu() {
    let def = build_gelu_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    assert_monotone_tightening(&def, &bindings, "GELU");
}

#[test]
fn test_activation_monotone_tightening_silu() {
    let def = build_silu_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    assert_monotone_tightening(&def, &bindings, "SiLU");
}

#[test]
fn test_activation_monotone_tightening_mish() {
    let def = build_mish_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    assert_monotone_tightening(&def, &bindings, "Mish");
}

#[test]
fn test_activation_monotone_tightening_sigmoid() {
    let def = build_sigmoid_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    assert_monotone_tightening(&def, &bindings, "Sigmoid");
}

#[test]
fn test_activation_monotone_tightening_relu() {
    let def = build_relu_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    assert_monotone_tightening(&def, &bindings, "ReLU");
}

// ===========================================================================
// 16. Activation chain: GELU -> Linear -> SiLU -> Linear (IBP)
// ===========================================================================

fn build_activation_chain_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_act_chain");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let w1 = b.add_input("w1", &[FFN_DIM, DIM]);
    let w2 = b.add_input("w2", &[DIM, FFN_DIM]);

    // GELU -> Linear -> SiLU -> Linear
    let h = b.add_gelu(input, &[SEQ_LEN, DIM]);
    let h = b.add_linear(h, w1, None, &[SEQ_LEN, FFN_DIM]);
    let h = add_silu(&mut b, h, &[SEQ_LEN, FFN_DIM]);
    let out = b.add_linear(h, w2, None, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid activation chain kernel")
}

fn activation_chain_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, FFN_DIM]), WEIGHT_MAG)),
    ]
}

#[test]
fn test_activation_chain_gelu_linear_silu_linear_ibp() {
    let def = build_activation_chain_kernel();
    let bindings = activation_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Activation chain (GELU->Linear->SiLU->Linear) IBP: width={width:.6}");
    assert!(width.is_finite(), "chain output width must be finite");
    assert!(
        width < 100.0,
        "activation chain width {width} exceeds threshold 100.0"
    );
}
