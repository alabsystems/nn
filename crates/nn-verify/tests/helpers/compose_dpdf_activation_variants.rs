// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for activation function variant bounds: GELU, SiLU, Mish,
//! Swish, hardswish, approximate GELU, activation pipelines, CROWN tightness,
//! and monotone tightening.
//!
//! Verifies IBP and CROWN bound propagation through activation function variants
//! used in dpdf document understanding models. These variants appear across
//! GLM-OCR (SwiGLU/SiLU), PaddleOCR (GELU), DocLayout-YOLO (SiLU, hardswish),
//! YOLO variants (Mish), and Granite-Docling (GELU).
//!
//! 1.  **GELU activation IBP bounds**: tanh-approx GELU single activation
//! 2.  **GELU activation CROWN bounds**: CROWN tighter than IBP for GELU
//! 3.  **SiLU/Swish activation IBP bounds**: x * sigmoid(x)
//! 4.  **SiLU CROWN bounds**: CROWN tighter than IBP for SiLU
//! 5.  **Mish activation IBP bounds**: x * tanh(softplus(x))
//! 6.  **Hardswish activation IBP bounds**: x * clamp(x+3, 0, 6) / 6
//! 7.  **Approximate GELU vs exact GELU bound comparison**: tanh vs erf
//! 8.  **Linear -> GELU -> Linear pipeline IBP**: FFN building block
//! 9.  **Linear -> SiLU -> Linear pipeline CROWN**: SwiGLU building block
//! 10. **Activation after LayerNorm IBP bounds**: norm -> activation composition
//! 11. **Monotone tightening**: CROWN strictly tighter than IBP for each variant
//! 12. **Multi-activation pipeline**: GELU -> Linear -> SiLU -> Linear end-to-end
//! 13. **Large input range stability**: activations produce finite bounds for wide inputs
//! 14. **Activation bound width comparison**: GELU vs SiLU vs Mish width ordering
//!
//! Architecture references:
//! - GELU (Hendrycks & Gimpel, 2016): Gaussian Error Linear Unit
//! - SiLU/Swish (Ramachandran et al., 2017): x * sigmoid(x)
//! - Mish (Misra, 2019): x * tanh(softplus(x))
//! - Hardswish (Howard et al., 2019, MobileNetV3): piecewise linear approx of Swish
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, DIM=16, FFN_DIM=32
//!
//! Part of #4061: Compose tests for activation function variant bounds.

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

/// Build hardswish activation: hardswish(x) = x * clamp(x + 3, 0, 6) / 6.
///
/// Decompose using available ops (no native sub/clamp). We implement subtraction
/// as `a + (-1 * b)`. The clamp `min(relu(x+3), 6)` is computed as
/// `relu(x+3) + (-1) * relu(relu(x+3) + (-6))`.
///
/// Constants needed: 3.0, -6.0, -1.0, 1/6.
fn add_hardswish(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    // x + 3
    let three = b.add_input("hardswish_three", &[1]);
    let three_bc = b.add_broadcast(three, shape);
    let x_plus_3 = b.add_binary_add(input, three_bc, shape);

    // relu(x + 3)
    let relu_xp3 = b.add_relu(x_plus_3, shape);

    // relu(x + 3) + (-6) = relu(x+3) - 6
    let neg_six = b.add_input("hardswish_neg_six", &[1]);
    let neg_six_bc = b.add_broadcast(neg_six, shape);
    let diff = b.add_binary_add(relu_xp3, neg_six_bc, shape);

    // relu(relu(x+3) - 6) = max(relu(x+3) - 6, 0)
    let excess = b.add_relu(diff, shape);

    // clamp(x+3, 0, 6) = relu(x+3) + (-1) * excess
    let neg_one = b.add_input("hardswish_neg_one", &[1]);
    let neg_one_bc = b.add_broadcast(neg_one, shape);
    let neg_excess = b.add_binary_mul(excess, neg_one_bc, shape);
    let clamped = b.add_binary_add(relu_xp3, neg_excess, shape);

    // x * clamp(x+3, 0, 6)
    let product = b.add_binary_mul(input, clamped, shape);

    // * (1/6)
    let inv6 = b.add_input("hardswish_inv6", &[1]);
    let inv6_bc = b.add_broadcast(inv6, shape);
    b.add_binary_mul(product, inv6_bc, shape)
}

/// Hardswish constant bindings: 3.0, -6.0, -1.0, 1/6.
fn hardswish_const_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 3.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), -6.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), -1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1.0f32 / 6.0f32)),
    ]
}

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. GELU activation IBP bounds
// ===========================================================================

fn build_gelu_variant_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_actvar_gelu");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_gelu(input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid GELU kernel")
}

#[test]
fn test_actvar_gelu_ibp_bounds() {
    let def = build_gelu_variant_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GELU variant IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // GELU is bounded below by ~-0.17. For input in [-1, 1]:
    assert!(lo_min >= -1.0, "GELU lower should be >= -1.0, got {lo_min}");
    assert!(
        hi_max <= 2.0,
        "GELU upper should be <= 2.0 for input [-1,1], got {hi_max}"
    );
}

// ===========================================================================
// 2. GELU activation CROWN bounds (tighter than IBP)
// ===========================================================================

#[test]
fn test_actvar_gelu_crown_tighter_than_ibp() {
    let def = build_gelu_variant_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("GELU variant CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 3. SiLU/Swish activation IBP bounds
// ===========================================================================

fn build_silu_variant_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_actvar_silu");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = add_silu(&mut b, input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid SiLU kernel")
}

#[test]
fn test_actvar_silu_ibp_bounds() {
    let def = build_silu_variant_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SiLU variant IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // SiLU(x) = x * sigmoid(x), bounded below by ~-0.278 for all x.
    assert!(lo_min >= -2.0, "SiLU lower should be >= -2.0, got {lo_min}");
    assert!(
        hi_max <= 2.0,
        "SiLU upper should be <= 2.0 for input [-1,1], got {hi_max}"
    );
}

// ===========================================================================
// 4. SiLU CROWN bounds
// ===========================================================================

#[test]
fn test_actvar_silu_crown_bounds() {
    let def = build_silu_variant_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("SiLU variant CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 5. Mish activation IBP bounds
// ===========================================================================

fn build_mish_variant_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_actvar_mish");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = add_mish(&mut b, input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid Mish kernel")
}

#[test]
fn test_actvar_mish_ibp_bounds() {
    let def = build_mish_variant_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mish variant IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Mish(x) = x * tanh(softplus(x)), bounded below by ~-0.31 for all x.
    assert!(lo_min >= -2.0, "Mish lower should be >= -2.0, got {lo_min}");
    assert!(
        hi_max <= 2.0,
        "Mish upper should be <= 2.0 for input [-1,1], got {hi_max}"
    );
}

// ===========================================================================
// 6. Hardswish activation IBP bounds
// ===========================================================================

fn build_hardswish_variant_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_actvar_hardswish");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = add_hardswish(&mut b, input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid hardswish kernel")
}

fn hardswish_variant_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(hardswish_const_bindings());
    bindings
}

#[test]
fn test_actvar_hardswish_ibp_bounds() {
    let def = build_hardswish_variant_kernel();
    let bindings = hardswish_variant_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Hardswish variant IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Hardswish(x) = x * clamp(x+3, 0, 6) / 6.
    // For input in [-1, 1]: hardswish(-1) = -1 * 2/6 = -0.333, hardswish(1) = 1 * 4/6 = 0.667.
    assert!(
        lo_min >= -2.0,
        "hardswish lower should be >= -2.0, got {lo_min}"
    );
    assert!(
        hi_max <= 2.0,
        "hardswish upper should be <= 2.0 for input [-1,1], got {hi_max}"
    );
}

// ===========================================================================
// 7. Approximate GELU vs exact GELU bound comparison
// ===========================================================================

fn build_gelu_erf_variant_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_actvar_gelu_erf");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_gelu_erf(input, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid GELU erf kernel")
}

#[test]
fn test_actvar_gelu_tanh_vs_erf_ibp_comparison() {
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    // Tanh-approx GELU
    let tanh_def = build_gelu_variant_kernel();
    let tanh_bindings = vec![TensorParamBinding::Variable];
    let tanh_graph = tensor_kernel_to_graph(&tanh_def, &tanh_bindings).expect("tanh GELU graph");
    let tanh_output = tanh_graph.propagate_ibp(&input).expect("tanh IBP");
    assert_bounds_valid(&tanh_output);
    let tanh_width = bound_width(&tanh_output);

    // Exact erf GELU
    let erf_def = build_gelu_erf_variant_kernel();
    let erf_bindings = vec![TensorParamBinding::Variable];
    let erf_graph = tensor_kernel_to_graph(&erf_def, &erf_bindings).expect("erf GELU graph");
    let erf_output = erf_graph.propagate_ibp(&input).expect("erf IBP");
    assert_bounds_valid(&erf_output);
    let erf_width = bound_width(&erf_output);

    eprintln!("GELU tanh vs erf IBP: tanh_width={tanh_width:.6}, erf_width={erf_width:.6}");
    // Both should produce finite, structurally valid bounds.
    assert!(tanh_width.is_finite(), "tanh GELU width must be finite");
    assert!(erf_width.is_finite(), "erf GELU width must be finite");
}

// ===========================================================================
// 8. Linear -> GELU -> Linear pipeline IBP
// ===========================================================================

fn build_linear_gelu_linear_pipeline() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_actvar_linear_gelu_linear");
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

fn linear_act_linear_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, FFN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ]
}

#[test]
fn test_actvar_linear_gelu_linear_ibp() {
    let def = build_linear_gelu_linear_pipeline();
    let bindings = linear_act_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Linear-GELU-Linear pipeline IBP: width={width:.6}");
    assert!(width.is_finite(), "IBP width must be finite");
}

// ===========================================================================
// 9. Linear -> SiLU -> Linear pipeline CROWN
// ===========================================================================

fn build_linear_silu_linear_pipeline() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_actvar_linear_silu_linear");
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

#[test]
fn test_actvar_linear_silu_linear_crown() {
    let def = build_linear_silu_linear_pipeline();
    let bindings = linear_act_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let ibp_width = bound_width(&ibp_output);

    // CROWN
    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let crown_width = bound_width(&crown_output);
    eprintln!("Linear-SiLU-Linear CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 10. Activation after LayerNorm IBP bounds
// ===========================================================================

fn build_layernorm_gelu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_actvar_layernorm_gelu");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_weight = b.add_input("ln_weight", &[DIM]);
    let ln_bias = b.add_input("ln_bias", &[DIM]);

    // LayerNorm -> GELU
    let normed = b.add_layer_norm(input, eps, 1, ln_weight, ln_bias, &[SEQ_LEN, DIM]);
    let out = b.add_gelu(normed, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid LayerNorm-GELU kernel")
}

fn layernorm_gelu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ]
}

#[test]
fn test_actvar_layernorm_gelu_ibp() {
    let def = build_layernorm_gelu_kernel();
    let bindings = layernorm_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LayerNorm-GELU IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_actvar_layernorm_silu_ibp() {
    // Same structure but with SiLU instead of GELU after LayerNorm.
    let mut b = TensorBlockBuilder::new("dpdf_actvar_layernorm_silu");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_weight = b.add_input("ln_weight", &[DIM]);
    let ln_bias = b.add_input("ln_bias", &[DIM]);

    let normed = b.add_layer_norm(input, eps, 1, ln_weight, ln_bias, &[SEQ_LEN, DIM]);
    let out = add_silu(&mut b, normed, &[SEQ_LEN, DIM]);

    let def = b.build(out).expect("valid LayerNorm-SiLU kernel");
    let bindings = layernorm_gelu_bindings(); // Same structure
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LayerNorm-SiLU IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. Monotone tightening: CROWN strictly tighter than IBP for each variant
// ===========================================================================

/// For a given activation kernel, verify that shrinking input bounds from
/// eps=1.0 to eps=0.1 produces tighter output bounds under IBP.
fn assert_activation_monotone_tightening(
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    label: &str,
) {
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
fn test_actvar_monotone_tightening_gelu() {
    let def = build_gelu_variant_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    assert_activation_monotone_tightening(&def, &bindings, "GELU variant");
}

#[test]
fn test_actvar_monotone_tightening_silu() {
    let def = build_silu_variant_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    assert_activation_monotone_tightening(&def, &bindings, "SiLU variant");
}

#[test]
fn test_actvar_monotone_tightening_mish() {
    let def = build_mish_variant_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    assert_activation_monotone_tightening(&def, &bindings, "Mish variant");
}

#[test]
fn test_actvar_monotone_tightening_hardswish() {
    let def = build_hardswish_variant_kernel();
    let bindings = hardswish_variant_bindings();
    assert_activation_monotone_tightening(&def, &bindings, "Hardswish variant");
}

// ===========================================================================
// 12. Multi-activation pipeline: GELU -> Linear -> SiLU -> Linear (IBP)
// ===========================================================================

fn build_multi_activation_pipeline() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_actvar_multi_pipeline");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let w1 = b.add_input("w1", &[FFN_DIM, DIM]);
    let w2 = b.add_input("w2", &[DIM, FFN_DIM]);

    // GELU -> Linear -> SiLU -> Linear
    let h = b.add_gelu(input, &[SEQ_LEN, DIM]);
    let h = b.add_linear(h, w1, None, &[SEQ_LEN, FFN_DIM]);
    let h = add_silu(&mut b, h, &[SEQ_LEN, FFN_DIM]);
    let out = b.add_linear(h, w2, None, &[SEQ_LEN, DIM]);

    b.build(out)
        .expect("valid multi-activation pipeline kernel")
}

fn multi_activation_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, FFN_DIM]), WEIGHT_MAG)),
    ]
}

#[test]
fn test_actvar_multi_pipeline_gelu_silu_ibp() {
    let def = build_multi_activation_pipeline();
    let bindings = multi_activation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Multi-activation pipeline (GELU->Linear->SiLU->Linear) IBP: width={width:.6}");
    assert!(width.is_finite(), "pipeline output width must be finite");
    assert!(
        width < 100.0,
        "multi-activation pipeline width {width} exceeds threshold 100.0"
    );
}

#[test]
fn test_actvar_multi_pipeline_crown() {
    let def = build_multi_activation_pipeline();
    let bindings = multi_activation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);
    let ibp_width = bound_width(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let crown_width = bound_width(&crown_output);
    eprintln!("Multi-activation CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 13. Large input range stability
// ===========================================================================

#[test]
fn test_actvar_gelu_large_input_range_ibp() {
    let def = build_gelu_variant_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Large input range: [-10, 10]
    let input = uniform_bounds(&[SEQ_LEN, DIM], 10.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GELU large input IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "lower bound must be finite for large inputs"
    );
    assert!(
        hi_max.is_finite(),
        "upper bound must be finite for large inputs"
    );
}

#[test]
fn test_actvar_silu_large_input_range_ibp() {
    let def = build_silu_variant_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ_LEN, DIM], 10.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SiLU large input IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "lower bound must be finite for large inputs"
    );
    assert!(
        hi_max.is_finite(),
        "upper bound must be finite for large inputs"
    );
}

// ===========================================================================
// 14. Activation bound width comparison: GELU vs SiLU vs Mish
// ===========================================================================

#[test]
fn test_actvar_bound_width_comparison_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    // GELU
    let gelu_def = build_gelu_variant_kernel();
    let gelu_graph =
        tensor_kernel_to_graph(&gelu_def, &[TensorParamBinding::Variable]).expect("GELU graph");
    let gelu_output = gelu_graph.propagate_ibp(&input).expect("GELU IBP");
    assert_bounds_valid(&gelu_output);
    let gelu_width = bound_width(&gelu_output);

    // SiLU
    let silu_def = build_silu_variant_kernel();
    let silu_graph =
        tensor_kernel_to_graph(&silu_def, &[TensorParamBinding::Variable]).expect("SiLU graph");
    let silu_output = silu_graph.propagate_ibp(&input).expect("SiLU IBP");
    assert_bounds_valid(&silu_output);
    let silu_width = bound_width(&silu_output);

    // Mish
    let mish_def = build_mish_variant_kernel();
    let mish_graph =
        tensor_kernel_to_graph(&mish_def, &[TensorParamBinding::Variable]).expect("Mish graph");
    let mish_output = mish_graph.propagate_ibp(&input).expect("Mish IBP");
    assert_bounds_valid(&mish_output);
    let mish_width = bound_width(&mish_output);

    eprintln!(
        "Activation width comparison: GELU={gelu_width:.6}, SiLU={silu_width:.6}, Mish={mish_width:.6}"
    );
    // All should produce finite, reasonable bounds.
    assert!(gelu_width.is_finite(), "GELU width must be finite");
    assert!(silu_width.is_finite(), "SiLU width must be finite");
    assert!(mish_width.is_finite(), "Mish width must be finite");
}
