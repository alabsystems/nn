// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for mixed-precision inference bounds (BF16/F16 compute).
//!
//! Verifies NY IBP and CROWN bound propagation through inference
//! pipelines using mixed-precision (BF16/F16) compute. Models reduced-precision
//! arithmetic as epsilon perturbations on weight magnitudes, capturing the
//! key verification property: how precision loss accumulates through depth
//! and interacts with nonlinear layers (softmax, normalization, attention).
//!
//! ## BF16/F16 Epsilon Model
//!
//! BF16 has 8-bit mantissa (vs FP32's 24-bit), giving ~3.9e-3 relative error.
//! F16 has 11-bit mantissa, giving ~4.88e-4 relative error.
//! We model reduced-precision weights as: w_reduced = w_fp32 * (1 ± eps_dtype)
//! where eps_dtype captures the worst-case rounding perturbation.
//!
//! ## Linear Layer BF16 Epsilon (tests 1-2)
//!
//! 1. Linear layer with BF16 epsilon perturbation IBP
//! 2. MatMul with reduced precision accumulation IBP
//!
//! ## Precision-Sensitive Operations (tests 3-6)
//!
//! 3. Softmax precision sensitivity at BF16 scale
//! 4. LayerNorm numerical stability at F16 precision
//! 5. RMSNorm with reduced precision IBP
//! 6. Attention score computation at BF16 IBP
//!
//! ## CROWN & Comparison (tests 7-8)
//!
//! 7. BF16 linear layer CROWN bounds
//! 8. F16 vs F32 bound width comparison
//!
//! ## Multi-Layer & Accumulation (tests 9-10)
//!
//! 9. Multi-layer precision loss accumulation IBP
//! 10. Mixed-precision FFN (BF16 compute, F32 accum) IBP
//!
//! ## Structural Properties (tests 11-13)
//!
//! 11. Monotone tightening for reduced-precision pipeline
//! 12. Full mixed-precision transformer block bounds
//! 13. Quantization error propagation through residual connections
//!
//! Mixed-precision inference scheme (BF16 compute / FP32 accumulate):
//!   Weights stored in BF16, activations computed in BF16, accumulated in FP32.
//!   Key risk: softmax and normalization layers amplify rounding error.
//!   BF16 machine epsilon: 2^-8 ≈ 3.9e-3 (8-bit mantissa).
//!   F16 machine epsilon: 2^-11 ≈ 4.88e-4 (11-bit mantissa).
//!
//! Architecture references:
//! - NVIDIA mixed-precision training (Micikevicius et al., 2018)
//! - BF16 inference on TPU/GPU for LLM serving
//! - Granite-Docling, Qwen3-VL: production BF16 inference
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, FFN_DIM=128
//!
//! Part of #4066: Compose tests for mixed-precision inference bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const SEQ_LEN: usize = 4;
const HIDDEN_DIM: usize = 64;
const FFN_DIM: usize = 128;
const NUM_HEADS: usize = 4;

// ---------------------------------------------------------------------------
// Precision parameters
// ---------------------------------------------------------------------------

/// FP32 baseline weight magnitude.
const FP32_WEIGHT_MAG: f32 = 0.02;

/// BF16 machine epsilon (2^-8): worst-case relative rounding error.
const BF16_EPS: f32 = 3.9e-3;

/// F16 machine epsilon (2^-11): worst-case relative rounding error.
const F16_EPS: f32 = 4.88e-4;

/// BF16 weight magnitude: FP32 weight * (1 + BF16_EPS) to model rounding.
const BF16_WEIGHT_MAG: f32 = FP32_WEIGHT_MAG * (1.0 + BF16_EPS);

/// F16 weight magnitude: FP32 weight * (1 + F16_EPS) to model rounding.
const F16_WEIGHT_MAG: f32 = FP32_WEIGHT_MAG * (1.0 + F16_EPS);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build SiLU activation: SiLU(x) = x * sigmoid(x).
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

/// Build a standard SwiGLU FFN block (graph topology only; weights via bindings).
fn build_swiglu_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> nn_dsl::TensorNodeId {
    let ffn_shape = [seq_len, ffn_dim];
    let out_shape = [seq_len, hidden_dim];

    let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[ffn_dim, hidden_dim]);
    let up_w = b.add_input(&format!("{prefix}_up_w"), &[ffn_dim, hidden_dim]);
    let down_w = b.add_input(&format!("{prefix}_down_w"), &[hidden_dim, ffn_dim]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(b, gate, &ffn_shape);

    let up = b.add_linear(input, up_w, None, &ffn_shape);

    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    b.add_linear(hidden, down_w, None, &out_shape)
}

/// Push SwiGLU weight bindings (gate_w, up_w, down_w) with given magnitude.
fn push_swiglu_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    hidden_dim: usize,
    ffn_dim: usize,
    weight_mag: f32,
) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ffn_dim, hidden_dim]),
        weight_mag,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ffn_dim, hidden_dim]),
        weight_mag,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[hidden_dim, ffn_dim]),
        weight_mag,
    )));
}

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Create a constant weight binding with given shape and magnitude.
fn weight_binding(shape: &[usize], mag: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), mag))
}

// ===========================================================================
// 1. Linear layer with BF16 epsilon perturbation IBP
// ===========================================================================

/// Verify that a linear layer with BF16-perturbed weights produces finite,
/// valid bounds. Models BF16 rounding as w_bf16 = w_fp32 * (1 + eps_bf16).
#[test]
fn test_bf16_linear_epsilon_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mixed_prec_bf16_linear");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w_bf16", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);
    let out = b.add_linear(input, w, Some(bias), &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid BF16 linear kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM], 0.01),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("BF16 linear epsilon IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. MatMul with reduced precision accumulation IBP
// ===========================================================================

/// Verify matmul bounds when accumulation is done in BF16 (no FP32 accumulator).
/// Without FP32 accumulation, rounding error is larger — modeled as
/// BF16_WEIGHT_MAG with an additional per-accumulation epsilon.
#[test]
fn test_bf16_matmul_reduced_accumulation_ibp() {
    // Model BF16 accumulation: each of HIDDEN_DIM additions contributes BF16_EPS
    // relative error, so effective weight magnitude is slightly larger.
    let accum_mag = BF16_WEIGHT_MAG * (1.0 + BF16_EPS * (HIDDEN_DIM as f32).sqrt());

    let mut b = TensorBlockBuilder::new("dpdf_mixed_prec_bf16_matmul");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w_accum", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid BF16 matmul kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], accum_mag),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("BF16 matmul reduced accum IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

// ===========================================================================
// 3. Softmax precision sensitivity at BF16 scale
// ===========================================================================

/// Softmax is precision-sensitive: exp() amplifies rounding errors.
/// Verify that BF16-precision logits -> softmax still produces valid
/// probability bounds in (0, 1).
#[test]
fn test_bf16_softmax_precision_sensitivity_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mixed_prec_bf16_softmax");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    // BF16-precision linear projection to logits
    let w = b.add_input("w_bf16", &[HIDDEN_DIM, HIDDEN_DIM]);
    let logits = b.add_linear(input, w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Softmax normalizes to probabilities
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid BF16 softmax kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("BF16 softmax precision IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "softmax output lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax output upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 4. LayerNorm numerical stability at F16 precision
// ===========================================================================

/// LayerNorm variance computation is precision-sensitive: F16 has limited
/// dynamic range. Verify that F16-magnitude weights through RMSNorm
/// (LayerNorm variant) produce stable, finite bounds.
#[test]
fn test_f16_layernorm_stability_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mixed_prec_f16_layernorm");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // RMSNorm with F16-precision weights
    let eps = b.add_input("eps", &[1]);
    let norm_weight = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_weight, &shape);

    // F16-precision linear after normalization
    let w = b.add_input("w_f16", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(normed, w, None, &shape);
    let def = b.build(out).expect("valid F16 LayerNorm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], F16_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("F16 LayerNorm stability IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

// ===========================================================================
// 5. RMSNorm with reduced precision IBP
// ===========================================================================

/// RMSNorm divides by sqrt(mean(x^2) + eps). At BF16 precision, the
/// reciprocal sqrt is less accurate. Verify bounds are finite and valid.
#[test]
fn test_bf16_rmsnorm_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mixed_prec_bf16_rmsnorm");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let eps = b.add_input("eps", &[1]);
    let norm_weight = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(input, eps, 1, norm_weight, &shape);
    let def = b.build(out).expect("valid BF16 RMSNorm kernel");

    // BF16-perturbed norm weight: 1.0 * (1 + BF16_EPS) to model rounding
    let bf16_norm_weight = 1.0 + BF16_EPS;
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM]),
            bf16_norm_weight,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("BF16 RMSNorm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. Attention score computation at BF16 IBP
// ===========================================================================

/// Multi-head attention with BF16-precision QKV projections.
/// Attention score = softmax(Q @ K^T / sqrt(d_k)) is precision-sensitive
/// because the dot product accumulates rounding errors across d_k dimensions.
#[test]
fn test_bf16_attention_scores_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mixed_prec_bf16_attention");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let attn_out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    let def = b.build(attn_out).expect("valid BF16 attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG), // q_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG), // k_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG), // v_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG), // o_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("BF16 attention scores IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. BF16 linear layer CROWN bounds
// ===========================================================================

fn build_bf16_linear_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mixed_prec_bf16_crown");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w_bf16", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);
    let out = b.add_linear(input, w, Some(bias), &[SEQ_LEN, HIDDEN_DIM]);
    b.build(out).expect("valid BF16 linear kernel")
}

fn bf16_linear_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM], 0.01),
    ]
}

#[test]
fn test_bf16_linear_crown() {
    let def = build_bf16_linear_kernel();
    let bindings = bf16_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("BF16 linear CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 8. F16 vs F32 bound width comparison
// ===========================================================================

/// Compare output bound widths of F16-precision vs FP32-precision linear.
/// F16 has slightly larger effective weight magnitude due to rounding,
/// which should produce slightly wider (or comparable) bounds.
#[test]
fn test_f16_vs_f32_bound_width_comparison_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mixed_prec_f16_f32_cmp");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid linear kernel");

    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // FP32 baseline
    let fp32_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG),
    ];
    let fp32_graph = tensor_kernel_to_graph(&def, &fp32_bindings).expect("FP32 graph");
    let fp32_output = fp32_graph.propagate_ibp(&input_bounds).expect("FP32 IBP");
    assert_bounds_valid(&fp32_output);
    let fp32_width = bound_width(&fp32_output);

    // F16 weights (slightly larger magnitude due to rounding)
    let f16_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], F16_WEIGHT_MAG),
    ];
    let f16_graph = tensor_kernel_to_graph(&def, &f16_bindings).expect("F16 graph");
    let f16_output = f16_graph.propagate_ibp(&input_bounds).expect("F16 IBP");
    assert_bounds_valid(&f16_output);
    let f16_width = bound_width(&f16_output);

    eprintln!("F16 vs F32 bound width IBP: f16_width={f16_width:.6}, fp32_width={fp32_width:.6}");
    // Both should produce finite, reasonable bounds
    assert!(f16_width.is_finite(), "F16 width must be finite");
    assert!(fp32_width.is_finite(), "FP32 width must be finite");
    // F16 has larger effective weight magnitude => wider or equal bounds
    assert!(
        f16_width >= fp32_width - 1e-4,
        "F16 bounds should be >= FP32: f16={f16_width}, fp32={fp32_width}"
    );
}

// ===========================================================================
// 9. Multi-layer precision loss accumulation IBP
// ===========================================================================

/// Verify that BF16 precision loss accumulates through multiple linear layers.
/// Each layer adds BF16_EPS relative error; after N layers the effective
/// weight magnitude grows. Compare 1-layer vs 3-layer bound widths.
#[test]
fn test_bf16_multi_layer_precision_accumulation_ibp() {
    // Build 1-layer graph
    let mut b1 = TensorBlockBuilder::new("dpdf_mixed_prec_bf16_1layer");
    let input1 = b1.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w1 = b1.add_input("w1", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out1 = b1.add_linear(input1, w1, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def1 = b1.build(out1).expect("valid 1-layer kernel");

    let bindings1 = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG),
    ];
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("1-layer graph");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output1 = graph1.propagate_ibp(&input_bounds).expect("1-layer IBP");
    assert_bounds_valid(&output1);
    let width1 = bound_width(&output1);

    // Build 3-layer graph
    let mut b3 = TensorBlockBuilder::new("dpdf_mixed_prec_bf16_3layer");
    let input3 = b3.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let wa = b3.add_input("w_a", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h1 = b3.add_linear(input3, wa, None, &shape);
    let wb = b3.add_input("w_b", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h2 = b3.add_linear(h1, wb, None, &shape);
    let wc = b3.add_input("w_c", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out3 = b3.add_linear(h2, wc, None, &shape);
    let def3 = b3.build(out3).expect("valid 3-layer kernel");

    let bindings3 = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG),
    ];
    let graph3 = tensor_kernel_to_graph(&def3, &bindings3).expect("3-layer graph");
    let output3 = graph3.propagate_ibp(&input_bounds).expect("3-layer IBP");
    assert_bounds_valid(&output3);
    let width3 = bound_width(&output3);

    eprintln!(
        "BF16 precision accumulation IBP: 1-layer width={width1:.6}, 3-layer width={width3:.6}"
    );
    // More layers => wider bounds (precision loss accumulates)
    assert!(
        width3 >= width1 - 1e-4,
        "3-layer should be wider: 3-layer={width3}, 1-layer={width1}"
    );
}

// ===========================================================================
// 10. Mixed-precision FFN (BF16 compute, F32 accum) IBP
// ===========================================================================

/// Mixed-precision SwiGLU FFN: BF16 weights for gate/up projections,
/// FP32-magnitude weights for the down projection (modeling FP32 accumulation).
#[test]
fn test_mixed_precision_ffn_bf16_fp32_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mixed_prec_ffn_bf16_fp32");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // BF16 gate and up projections
    let gate_w = b.add_input("gate_w", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, HIDDEN_DIM]);
    // FP32 down projection (accumulation in full precision)
    let down_w = b.add_input("down_w", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);

    let up = b.add_linear(input, up_w, None, &ffn_shape);

    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &out_shape);
    let def = b.build(out).expect("valid mixed FFN kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[FFN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG), // gate: BF16
        weight_binding(&[FFN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG), // up: BF16
        weight_binding(&[HIDDEN_DIM, FFN_DIM], FP32_WEIGHT_MAG), // down: FP32
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Mixed-precision FFN (BF16 compute, FP32 accum) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. Monotone tightening for reduced-precision pipeline
// ===========================================================================

/// Verify monotone tightening: tighter input bounds -> tighter output bounds
/// for a BF16-precision RMSNorm + linear pipeline.
#[test]
fn test_bf16_pipeline_monotone_tightening_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mixed_prec_bf16_monotone");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // RMSNorm
    let eps = b.add_input("eps", &[1]);
    let norm_weight = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_weight, &shape);

    // BF16 linear
    let w = b.add_input("w_bf16", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(normed, w, None, &shape);
    let def = b.build(out).expect("valid monotone kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input
    let wide_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("wide IBP");
    assert_bounds_valid(&wide_output);
    let wide_width = bound_width(&wide_output);

    // Narrow input
    let narrow_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("narrow IBP");
    assert_bounds_valid(&narrow_output);
    let narrow_width = bound_width(&narrow_output);

    eprintln!(
        "BF16 monotone tightening IBP: wide_width={wide_width:.6}, narrow_width={narrow_width:.6}"
    );
    // Tighter input should produce tighter or equal output
    assert!(
        narrow_width <= wide_width + 1e-4,
        "narrow output should be tighter: narrow={narrow_width}, wide={wide_width}"
    );
}

// ===========================================================================
// 12. Full mixed-precision transformer block bounds
// ===========================================================================

fn build_mixed_precision_transformer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mixed_prec_transformer");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Pre-norm: RMSNorm before attention
    let attn_eps = b.add_input("attn_eps", &[1]);
    let attn_norm_w = b.add_input("attn_norm_w", &[HIDDEN_DIM]);
    let normed_attn = b.add_rms_norm(input, attn_eps, 1, attn_norm_w, &shape);

    // BF16 attention projections
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let attn_out = b
        .add_multi_head_attention(
            normed_attn,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // First residual
    let h = b.add_binary_add(input, attn_out, &shape);

    // Pre-norm: RMSNorm before FFN
    let ffn_eps = b.add_input("ffn_eps", &[1]);
    let ffn_norm_w = b.add_input("ffn_norm_w", &[HIDDEN_DIM]);
    let normed_ffn = b.add_rms_norm(h, ffn_eps, 1, ffn_norm_w, &shape);

    // Mixed-precision SwiGLU FFN (BF16 compute)
    let ffn_out = build_swiglu_block(&mut b, normed_ffn, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Second residual
    let out = b.add_binary_add(h, ffn_out, &shape);
    b.build(out)
        .expect("valid mixed-precision transformer kernel")
}

fn mixed_precision_transformer_bindings() -> Vec<TensorParamBinding> {
    let eps = TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32));
    let norm_w =
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32));

    let mut bindings = vec![
        TensorParamBinding::Variable,
        eps.clone(),                                                // attn_eps
        norm_w.clone(),                                             // attn_norm_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG), // q_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG), // k_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG), // v_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG), // o_w
        eps,                                                        // ffn_eps
        norm_w,                                                     // ffn_norm_w
    ];
    // BF16 SwiGLU weights
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, BF16_WEIGHT_MAG);
    bindings
}

#[test]
fn test_full_mixed_precision_transformer_ibp() {
    let def = build_mixed_precision_transformer_kernel();
    let bindings = mixed_precision_transformer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full mixed-precision transformer IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_full_mixed_precision_transformer_crown() {
    let def = build_mixed_precision_transformer_kernel();
    let bindings = mixed_precision_transformer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Full mixed-precision transformer CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 13. Quantization error propagation through residual connections
// ===========================================================================

/// Residual connections: x + Linear_BF16(x). The skip connection preserves
/// the original FP32-range input while adding BF16-precision projection output.
/// This models the common mixed-precision pattern where activations on the
/// residual stream remain in FP32 while sub-blocks compute in BF16.
#[test]
fn test_bf16_residual_error_propagation_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mixed_prec_bf16_residual");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // First residual: x + BF16_Linear(x)
    let w1 = b.add_input("w1_bf16", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj1 = b.add_linear(input, w1, None, &shape);
    let h = b.add_binary_add(input, proj1, &shape);

    // Second residual: h + BF16_Linear(h)
    let w2 = b.add_input("w2_bf16", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj2 = b.add_linear(h, w2, None, &shape);
    let out = b.add_binary_add(h, proj2, &shape);

    let def = b.build(out).expect("valid BF16 residual kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], BF16_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("BF16 residual error propagation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Residual preserves input range plus BF16 projection, should be bounded
    assert!(
        lo_min > -100.0,
        "residual lower should be reasonable, got {lo_min}"
    );
}
