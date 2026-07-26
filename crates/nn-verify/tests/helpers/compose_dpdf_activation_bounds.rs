// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests proving activation function bounds propagate
//! correctly through multi-layer dpdf model architectures.
//!
//! Focuses on **bound correctness** for individual activations and their
//! composition with linear layers, normalization, residual connections,
//! softmax, and gated units. These patterns appear throughout dpdf document
//! understanding models: GELU in transformer MLPs, SiLU/SwiGLU in GLM-OCR
//! and Qwen3-VL, ReLU in Table Transformer backbone, Mish in YOLO variants,
//! sigmoid/tanh in detection heads and LSTM gates.
//!
//! ## Tests (18 tests):
//!
//! 1. **ReLU clips negative values (IBP)**: output lower bound >= 0 for
//!    input in [-1, 1].
//!
//! 2. **ReLU preserves positive upper bound (IBP)**: output upper bound
//!    does not exceed input upper bound for input in [0, 2].
//!
//! 3. **GELU single layer bounds (IBP)**: GELU on [-1, 1] input, output
//!    bounded within expected range.
//!
//! 4. **GELU through Linear + GELU (IBP + CROWN)**: transformer MLP
//!    pattern, verifies bound propagation through composition.
//!
//! 5. **SiLU bounds for [-2, 2] input (IBP)**: wider input range, SiLU
//!    output bounded below by ~-0.278.
//!
//! 6. **SiLU through feedforward network (IBP + CROWN)**: Linear -> SiLU ->
//!    Linear composition with IBP and CROWN comparison.
//!
//! 7. **Mish bounds for detection head (IBP)**: Mish activation in
//!    detection context, bounded below by ~-0.31.
//!
//! 8. **Sigmoid bounds always in (0, 1) (IBP)**: fundamental sigmoid
//!    property verified under IBP.
//!
//! 9. **Tanh bounds always in (-1, 1) (IBP)**: fundamental tanh property
//!    verified under IBP.
//!
//! 10. **Activation after LayerNorm (IBP)**: LayerNorm -> GELU combined
//!     bounds propagation.
//!
//! 11. **Activation before linear projection (IBP)**: ReLU -> Linear,
//!     verifies ReLU non-negativity feeds into linear projection.
//!
//! 12. **Stacked Linear + GELU + Linear 2-layer (IBP)**: two consecutive
//!     MLP blocks with GELU, verifies finite bound propagation.
//!
//! 13. **Stacked Linear + GELU + Linear 4-layer (IBP)**: deeper 4-block
//!     MLP chain, verifies bounds remain finite through depth.
//!
//! 14. **Mixed activations: ReLU backbone + GELU head (IBP)**: backbone
//!     uses ReLU, head uses GELU, verifies cross-activation composition.
//!
//! 15. **Activation with residual connection (IBP)**: x + GELU(Linear(x))
//!     residual pattern, verifies residual does not collapse bounds.
//!
//! 16. **Softmax after activation (IBP)**: GELU -> Linear -> Softmax,
//!     verifies output is valid probability distribution in [0, 1].
//!
//! 17. **Activation gradient bounds for training (IBP)**: verifies the
//!     sigmoid-derivative product graph sig*(1-sig). The function maximum is
//!     0.25, but plain IBP through the explicit product cannot track
//!     s+(1-s)=1, so for x in [-3,3] the IBP-correct upper bound is
//!     sigmoid(3)^2 ~= 0.9074 (the function's 0.25 bound needs CROWN).
//!
//! 18. **GLU / SwiGLU gated activation bounds (IBP)**: GLU splits input
//!     in half, applies sigmoid gate; SwiGLU uses SiLU gate. Both verified.
//!
//! Dimensions (small for fast verification):
//! - SEQ_LEN=4, DIM=16, FFN_DIM=32
//!
//! Part of #4118: Compose tests for activation function bounds propagation.

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

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Constant zero bias tensor binding.
fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Norm weight (all ones) binding.
fn norm_weight(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 1.0f32))
}

/// Norm bias (all zeros) binding.
fn norm_bias(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0f32))
}

/// Epsilon scalar binding for normalization.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
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

/// Build Mish activation: Mish(x) = x * tanh(softplus(x)).
fn add_mish(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sp = b.add_softplus(input, shape);
    let th = b.add_tanh(sp, shape);
    b.add_binary_mul(input, th, shape)
}

// ===========================================================================
// 1. ReLU clips negative values: output lower bound >= 0
// ===========================================================================

#[test]
fn test_relu_clips_negative_lower_bound() {
    let mut b = TensorBlockBuilder::new("act_bounds_relu_clip");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_relu(input, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid ReLU kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0); // [-1, 1]

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ReLU clip: bounds=[{lo_min:.6}, {hi_max:.6}]");
    let tol = 1e-6;
    assert!(
        lo_min >= 0.0 - tol,
        "ReLU must clip negatives: lower bound {lo_min} should be >= 0"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "ReLU upper must not exceed input upper for [-1,1]: got {hi_max}"
    );
}

// ===========================================================================
// 2. ReLU preserves positive upper bound
// ===========================================================================

#[test]
fn test_relu_preserves_positive_upper_bound() {
    let mut b = TensorBlockBuilder::new("act_bounds_relu_pos");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_relu(input, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid ReLU kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Input in [0, 2]: entirely non-negative
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, DIM]), 2.0f32),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ReLU positive: bounds=[{lo_min:.6}, {hi_max:.6}]");
    let tol = 1e-6;
    assert!(lo_min >= 0.0 - tol, "ReLU lower must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 2.0 + tol,
        "ReLU must preserve positive upper bound: got {hi_max}, expected <= 2.0"
    );
}

// ===========================================================================
// 3. GELU single layer bounds for [-1, 1] input
// ===========================================================================

#[test]
fn test_gelu_single_layer_bounds() {
    let mut b = TensorBlockBuilder::new("act_bounds_gelu_single");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_gelu(input, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid GELU kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0); // [-1, 1]

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GELU single: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // GELU is bounded below by ~-0.17 for all inputs.
    // For [-1, 1], output in approximately [-0.17, 1.0].
    assert!(lo_min >= -1.0, "GELU lower should be >= -1.0, got {lo_min}");
    assert!(
        hi_max <= 2.0,
        "GELU upper should be <= 2.0 for [-1,1] input, got {hi_max}"
    );
}

// ===========================================================================
// 4. GELU through Linear + GELU (transformer MLP pattern)
// ===========================================================================

fn build_linear_gelu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("act_bounds_linear_gelu");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let w = b.add_input("w", &[FFN_DIM, DIM]);
    let bias = b.add_input("bias", &[FFN_DIM]);

    let h = b.add_linear(input, w, Some(bias), &[SEQ_LEN, FFN_DIM]);
    let out = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);

    b.build(out).expect("valid Linear-GELU kernel")
}

fn linear_gelu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, DIM]),
        bias_zero(&[FFN_DIM]),
    ]
}

#[test]
fn test_gelu_through_linear_ibp_and_crown() {
    let def = build_linear_gelu_kernel();
    let bindings = linear_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let ibp_width = bound_width(&ibp_output);
    eprintln!("Linear+GELU IBP width: {ibp_width:.6}");
    assert!(ibp_width.is_finite(), "IBP width must be finite");

    // CROWN
    let (_method, crown_output, _fallback) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);

    let crown_width = bound_width(&crown_output);
    eprintln!("Linear+GELU CROWN width: {crown_width:.6}");
}

// ===========================================================================
// 5. SiLU bounds for [-2, 2] input range
// ===========================================================================

#[test]
fn test_silu_bounds_wide_input() {
    let mut b = TensorBlockBuilder::new("act_bounds_silu_wide");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = add_silu(&mut b, input, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid SiLU kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 2.0); // [-2, 2]

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SiLU wide [-2,2]: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // SiLU(x) = x * sigmoid(x), bounded below by ~-0.278 for all x.
    assert!(lo_min >= -3.0, "SiLU lower should be >= -3.0, got {lo_min}");
    assert!(
        hi_max <= 3.0,
        "SiLU upper should be <= 3.0 for [-2,2] input, got {hi_max}"
    );
}

// ===========================================================================
// 6. SiLU through feedforward network (IBP + CROWN)
// ===========================================================================

fn build_silu_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("act_bounds_silu_ffn");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let w1 = b.add_input("w1", &[FFN_DIM, DIM]);
    let b1 = b.add_input("b1", &[FFN_DIM]);
    let w2 = b.add_input("w2", &[DIM, FFN_DIM]);
    let b2 = b.add_input("b2", &[DIM]);

    let h = b.add_linear(input, w1, Some(b1), &[SEQ_LEN, FFN_DIM]);
    let h = add_silu(&mut b, h, &[SEQ_LEN, FFN_DIM]);
    let out = b.add_linear(h, w2, Some(b2), &[SEQ_LEN, DIM]);

    b.build(out).expect("valid SiLU FFN kernel")
}

fn silu_ffn_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, DIM]),
        bias_zero(&[FFN_DIM]),
        weight(&[DIM, FFN_DIM]),
        bias_zero(&[DIM]),
    ]
}

#[test]
fn test_silu_feedforward_ibp_and_crown() {
    let def = build_silu_ffn_kernel();
    let bindings = silu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let ibp_width = bound_width(&ibp_output);
    eprintln!("SiLU FFN IBP width: {ibp_width:.6}");
    assert!(ibp_width.is_finite(), "SiLU FFN IBP width must be finite");

    // CROWN
    let (_method, crown_output, _fallback) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);

    let crown_width = bound_width(&crown_output);
    eprintln!("SiLU FFN CROWN width: {crown_width:.6}");
}

// ===========================================================================
// 7. Mish bounds for detection head activation
// ===========================================================================

#[test]
fn test_mish_detection_head_bounds() {
    let mut b = TensorBlockBuilder::new("act_bounds_mish_det");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = add_mish(&mut b, input, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid Mish kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0); // [-1, 1]

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mish detection: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Mish(x) = x * tanh(softplus(x)), bounded below by ~-0.31.
    assert!(lo_min >= -2.0, "Mish lower should be >= -2.0, got {lo_min}");
    assert!(
        hi_max <= 2.0,
        "Mish upper should be <= 2.0 for [-1,1] input, got {hi_max}"
    );
}

// ===========================================================================
// 8. Sigmoid bounds: always in (0, 1)
// ===========================================================================

#[test]
fn test_sigmoid_always_bounded_01() {
    let mut b = TensorBlockBuilder::new("act_bounds_sigmoid_01");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_sigmoid(input, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid sigmoid kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Wide input range [-5, 5] to test saturation.
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 5.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sigmoid: bounds=[{lo_min:.6}, {hi_max:.6}]");
    let tol = 1e-6;
    assert!(
        lo_min >= 0.0 - tol,
        "sigmoid must have lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "sigmoid must have upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 9. Tanh bounds: always in (-1, 1)
// ===========================================================================

#[test]
fn test_tanh_always_bounded_neg1_1() {
    let mut b = TensorBlockBuilder::new("act_bounds_tanh");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let out = b.add_tanh(input, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid tanh kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Wide input range [-5, 5] to test saturation.
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 5.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Tanh: bounds=[{lo_min:.6}, {hi_max:.6}]");
    let tol = 1e-6;
    assert!(
        lo_min >= -1.0 - tol,
        "tanh must have lower >= -1, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "tanh must have upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 10. Activation after LayerNorm: combined bounds
// ===========================================================================

fn build_layernorm_gelu_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, DIM];
    let mut b = TensorBlockBuilder::new("act_bounds_ln_gelu");
    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[DIM]);
    let ln_b = b.add_input("ln_bias", &[DIM]);

    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);
    let out = b.add_gelu(normed, &shape);

    b.build(out).expect("valid LayerNorm-GELU kernel")
}

fn layernorm_gelu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        norm_weight(DIM),
        norm_bias(DIM),
    ]
}

#[test]
fn test_activation_after_layernorm() {
    let def = build_layernorm_gelu_kernel();
    let bindings = layernorm_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LayerNorm+GELU: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // LayerNorm normalizes, then GELU clips below ~-0.17.
    // Output should be bounded.
    let width = hi_max - lo_min;
    assert!(
        width < 50.0,
        "LayerNorm+GELU output width {width} should be bounded"
    );
}

// ===========================================================================
// 11. Activation before linear projection: bounds propagation
// ===========================================================================

fn build_relu_linear_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("act_bounds_relu_linear");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let w = b.add_input("w", &[DIM, DIM]);

    let activated = b.add_relu(input, &[SEQ_LEN, DIM]);
    let out = b.add_linear(activated, w, None, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid ReLU-Linear kernel")
}

fn relu_linear_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable, weight(&[DIM, DIM])]
}

#[test]
fn test_activation_before_linear_projection() {
    let def = build_relu_linear_kernel();
    let bindings = relu_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("ReLU+Linear: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    assert!(width.is_finite(), "ReLU+Linear width must be finite");
    // ReLU clips to [0, 1], then linear with small weights should not blow up.
    assert!(
        width < 20.0,
        "ReLU+Linear width {width} should be reasonable"
    );
}

// ===========================================================================
// 12. Stacked Linear + GELU + Linear (2-layer MLP)
// ===========================================================================

fn build_stacked_gelu_mlp_kernel(n_layers: usize) -> TensorKernelDef {
    let name = format!("act_bounds_stacked_gelu_{n_layers}layer");
    let mut b = TensorBlockBuilder::new(&name);
    let mut current = b.add_input("hidden", &[SEQ_LEN, DIM]);

    for i in 0..n_layers {
        let w1 = b.add_input(&format!("w1_{i}"), &[FFN_DIM, DIM]);
        let b1 = b.add_input(&format!("b1_{i}"), &[FFN_DIM]);
        let w2 = b.add_input(&format!("w2_{i}"), &[DIM, FFN_DIM]);
        let b2 = b.add_input(&format!("b2_{i}"), &[DIM]);

        let h = b.add_linear(current, w1, Some(b1), &[SEQ_LEN, FFN_DIM]);
        let h = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
        current = b.add_linear(h, w2, Some(b2), &[SEQ_LEN, DIM]);
    }

    b.build(current).expect("valid stacked GELU MLP kernel")
}

fn stacked_gelu_mlp_bindings(n_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _ in 0..n_layers {
        bindings.push(weight(&[FFN_DIM, DIM])); // w1
        bindings.push(bias_zero(&[FFN_DIM])); // b1
        bindings.push(weight(&[DIM, FFN_DIM])); // w2
        bindings.push(bias_zero(&[DIM])); // b2
    }
    bindings
}

#[test]
fn test_stacked_gelu_mlp_2_layer() {
    let def = build_stacked_gelu_mlp_kernel(2);
    let bindings = stacked_gelu_mlp_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("Stacked GELU MLP 2-layer: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    assert!(lo_min.is_finite(), "lower bound must be finite at depth=2");
    assert!(hi_max.is_finite(), "upper bound must be finite at depth=2");
}

// ===========================================================================
// 13. Stacked Linear + GELU + Linear (4-layer MLP)
// ===========================================================================

#[test]
fn test_stacked_gelu_mlp_4_layer() {
    let def = build_stacked_gelu_mlp_kernel(4);
    let bindings = stacked_gelu_mlp_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("Stacked GELU MLP 4-layer: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    assert!(lo_min.is_finite(), "lower bound must be finite at depth=4");
    assert!(hi_max.is_finite(), "upper bound must be finite at depth=4");
}

// ===========================================================================
// 14. Mixed activations: ReLU in backbone + GELU in head
// ===========================================================================

fn build_mixed_activation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("act_bounds_mixed_relu_gelu");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);

    // Backbone: Linear -> ReLU -> Linear (ReLU backbone pattern)
    let w_bb1 = b.add_input("backbone_w1", &[FFN_DIM, DIM]);
    let w_bb2 = b.add_input("backbone_w2", &[DIM, FFN_DIM]);

    let h = b.add_linear(input, w_bb1, None, &[SEQ_LEN, FFN_DIM]);
    let h = b.add_relu(h, &[SEQ_LEN, FFN_DIM]);
    let backbone_out = b.add_linear(h, w_bb2, None, &[SEQ_LEN, DIM]);

    // Head: Linear -> GELU -> Linear (transformer head pattern)
    let w_hd1 = b.add_input("head_w1", &[FFN_DIM, DIM]);
    let w_hd2 = b.add_input("head_w2", &[DIM, FFN_DIM]);

    let h = b.add_linear(backbone_out, w_hd1, None, &[SEQ_LEN, FFN_DIM]);
    let h = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
    let out = b.add_linear(h, w_hd2, None, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid mixed activation kernel")
}

fn mixed_activation_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, DIM]), // backbone_w1
        weight(&[DIM, FFN_DIM]), // backbone_w2
        weight(&[FFN_DIM, DIM]), // head_w1
        weight(&[DIM, FFN_DIM]), // head_w2
    ]
}

#[test]
fn test_mixed_activations_relu_backbone_gelu_head() {
    let def = build_mixed_activation_kernel();
    let bindings = mixed_activation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("Mixed ReLU+GELU: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        width < 100.0,
        "mixed activation width {width} should be reasonable"
    );
}

// ===========================================================================
// 15. Activation with residual connection bounds
// ===========================================================================

fn build_residual_gelu_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, DIM];
    let mut b = TensorBlockBuilder::new("act_bounds_residual_gelu");
    let input = b.add_input("hidden", &shape);
    let w = b.add_input("w", &[DIM, DIM]);
    let bias = b.add_input("bias", &[DIM]);

    // Residual: x + GELU(Linear(x))
    let h = b.add_linear(input, w, Some(bias), &shape);
    let h_act = b.add_gelu(h, &shape);
    let out = b.add_binary_add(input, h_act, &shape);

    b.build(out).expect("valid residual GELU kernel")
}

fn residual_gelu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[DIM, DIM]),
        bias_zero(&[DIM]),
    ]
}

#[test]
fn test_activation_with_residual_connection() {
    let def = build_residual_gelu_kernel();
    let bindings = residual_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("Residual+GELU: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    // Residual adds input bounds to sublayer bounds. Width should be at least
    // as wide as input (2.0 for eps=1.0).
    assert!(
        width >= 1.5,
        "residual width should be at least ~input width, got {width}"
    );
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 16. Softmax after activation: probability bounds
// ===========================================================================

fn build_gelu_linear_softmax_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("act_bounds_gelu_softmax");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let w = b.add_input("w", &[DIM, DIM]);

    // GELU -> Linear -> Softmax
    let h = b.add_gelu(input, &[SEQ_LEN, DIM]);
    let h = b.add_linear(h, w, None, &[SEQ_LEN, DIM]);
    let out = b.add_softmax(h, -1, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid GELU-Softmax kernel")
}

fn gelu_linear_softmax_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable, weight(&[DIM, DIM])]
}

#[test]
fn test_softmax_after_activation_probability_bounds() {
    let def = build_gelu_linear_softmax_kernel();
    let bindings = gelu_linear_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GELU+Softmax: bounds=[{lo_min:.6}, {hi_max:.6}]");
    let tol = 1e-5;
    assert!(
        lo_min >= 0.0 - tol,
        "softmax output must have lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax output must have upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 17. Activation gradient bounds (for training verification)
// ===========================================================================
//
// Sigmoid derivative: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)).
// This is bounded in (0, 0.25] for all x. We verify this property by
// composing sigmoid(x) * (1 - sigmoid(x)) and checking IBP bounds.

fn build_sigmoid_derivative_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, DIM];
    let mut b = TensorBlockBuilder::new("act_bounds_sigmoid_grad");
    let input = b.add_input("x", &shape);

    // sigmoid(x)
    let sig = b.add_sigmoid(input, &shape);

    // 1 - sigmoid(x): use broadcast constant 1.0 and subtract via
    // add(-sig) + 1.0 pattern. But TensorBlockBuilder has no sub; use
    // the identity: 1 - sig = (1_const + (-1)*sig).
    // Simpler approach: negate via mul(-1) then add 1.
    let neg_one = b.add_input("neg_one", &[1]);
    let neg_one_bc = b.add_broadcast(neg_one, &shape);
    let neg_sig = b.add_binary_mul(sig, neg_one_bc, &shape);
    let one = b.add_input("one", &[1]);
    let one_bc = b.add_broadcast(one, &shape);
    let one_minus_sig = b.add_binary_add(one_bc, neg_sig, &shape);

    // sigmoid'(x) = sig * (1 - sig)
    let out = b.add_binary_mul(sig, one_minus_sig, &shape);

    b.build(out).expect("valid sigmoid derivative kernel")
}

fn sigmoid_derivative_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(-1.0), // neg_one
        TensorParamBinding::ConstantScalar(1.0),  // one
    ]
}

#[test]
fn test_activation_gradient_bounds_sigmoid_derivative() {
    let def = build_sigmoid_derivative_kernel();
    let bindings = sigmoid_derivative_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 3.0); // [-3, 3]

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sigmoid derivative: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // The *function* sigmoid'(x) = sig(x) * (1-sig(x)) is in (0, 0.25] for all x.
    // But this graph computes it as an explicit product of two separately-bounded
    // intervals, and plain IBP cannot track the correlation s + (1-s) = 1. With
    // x in [-3, 3], IBP gives sig in [sigmoid(-3), sigmoid(3)] = [0.04743, 0.95257]
    // and 1-sig in the same interval, so the product upper bound is
    // sigmoid(3)^2 = 0.95257^2 ~= 0.9074. That is the IBP-correct bound of the
    // product GRAPH, not the analytic 0.25 maximum of the function. (CROWN would
    // be needed to recover a 0.25-class bound.)
    let tol = 0.01; // small slack for the f32 sigmoid evaluation
    assert!(
        lo_min >= 0.0 - tol,
        "sigmoid derivative must be >= 0, got {lo_min}"
    );
    // IBP product upper bound: sigmoid(3)^2 ~= 0.9074 (+ slack).
    assert!(
        hi_max <= 0.9074 + tol,
        "sigmoid derivative IBP product bound must be <= sigmoid(3)^2 ~= 0.9074 (+ slack), got {hi_max}"
    );
}

// ===========================================================================
// 18. GLU / SwiGLU gated activation bounds
// ===========================================================================
//
// GLU: splits input along last axis, applies sigmoid gate to second half,
//      multiplies with first half. Output bounded by (0, 1) gating.
// SwiGLU: gate_proj -> SiLU, up_proj -> element-wise mul, down_proj.

fn build_glu_kernel() -> TensorKernelDef {
    // GLU needs input dim to be 2*DIM (split in half).
    let double_dim = DIM * 2;
    let mut b = TensorBlockBuilder::new("act_bounds_glu");
    let input = b.add_input("x", &[SEQ_LEN, double_dim]);

    let out = b
        .add_glu(input, 1, &[SEQ_LEN, double_dim])
        .expect("valid GLU");

    b.build(out).expect("valid GLU kernel")
}

fn build_swiglu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("act_bounds_swiglu");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, DIM]);
    let down_w = b.add_input("down_weight", &[DIM, FFN_DIM]);

    // SwiGLU: gate_proj -> SiLU, up_proj, mul, down_proj
    let gate = b.add_linear(input, gate_w, None, &[SEQ_LEN, FFN_DIM]);
    let gate_act = add_silu(&mut b, gate, &[SEQ_LEN, FFN_DIM]);
    let up = b.add_linear(input, up_w, None, &[SEQ_LEN, FFN_DIM]);
    let gated = b.add_binary_mul(gate_act, up, &[SEQ_LEN, FFN_DIM]);
    let out = b.add_linear(gated, down_w, None, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid SwiGLU kernel")
}

fn swiglu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, DIM]), // gate_weight
        weight(&[FFN_DIM, DIM]), // up_weight
        weight(&[DIM, FFN_DIM]), // down_weight
    ]
}

#[test]
fn test_glu_gated_activation_bounds() {
    let double_dim = DIM * 2;
    let def = build_glu_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, double_dim], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("GLU: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    // GLU output = data * sigmoid(gate). Since sigmoid is in (0,1) and
    // data is in [-1, 1], output should be bounded within [-1, 1].
    assert!(width.is_finite(), "GLU width must be finite");
    assert!(
        width < 5.0,
        "GLU output width {width} should be bounded (sigmoid gating)"
    );
}

#[test]
fn test_swiglu_gated_activation_bounds() {
    let def = build_swiglu_kernel();
    let bindings = swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("SwiGLU: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    assert!(width.is_finite(), "SwiGLU width must be finite");
    assert!(
        width < 50.0,
        "SwiGLU output width {width} should be bounded"
    );
}
