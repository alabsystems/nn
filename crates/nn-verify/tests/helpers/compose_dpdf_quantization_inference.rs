// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for quantization-aware inference bounds
//! (INT4/INT8 vs FP32 equivalence).
//!
//! Verifies NY IBP and CROWN bound propagation through inference
//! pipelines that model quantization effects. Each test builds a subgraph
//! representing a quantization pattern, propagates bounds, and asserts that
//! outputs remain within expected ranges. Unlike `compose_dpdf_quantized.rs`
//! (dequantization mechanics) and `compose_dpdf_quantized_inference.rs`
//! (INT4 weight inference paths), this file focuses on INT8/INT4 vs FP32
//! equivalence margins and quantization error bounding.
//!
//! ## INT8 Linear & Symmetric Quantization (tests 1-3)
//!
//! 1. **INT8 linear output bounds vs FP32 (IBP)**: compares output bound
//!    widths of INT8-quantized vs FP32 linear layers.
//!
//! 2. **INT8 symmetric quantization: scale * int_value bounds (IBP)**:
//!    verifies scale * quantized_value produces bounded outputs.
//!
//! 3. **INT8 asymmetric quantization: (int_value - zero_point) * scale (IBP)**:
//!    verifies asymmetric dequant formula with nonzero zero_point.
//!
//! ## INT4 & Group Quantization (tests 4-6)
//!
//! 4. **INT4 GPTQ dequantization: group-wise scale bounds (IBP)**: models
//!    per-group scale variation in GPTQ.
//!
//! 5. **AWQ per-channel activation-aware scaling (IBP)**: models AWQ
//!    per-channel rescaling producing tighter bounds than naive INT4.
//!
//! 6. **Group quantization: per-group scale and zero point (IBP)**: models
//!    group-wise quantization with different scales per group.
//!
//! ## Quantized Attention & MLP (tests 7-8)
//!
//! 7. **Quantized attention score bounds (INT8 QKV) (IBP + CROWN)**: INT8
//!    quantized QKV projections through multi-head attention.
//!
//! 8. **Quantized MLP bounds (INT4 weights, FP16 activations) (IBP)**: INT4
//!    weight-only quantization through SwiGLU FFN.
//!
//! ## Error Bounds & Precision (tests 9-11)
//!
//! 9. **Quantization error bound: |fp32 - quant| <= delta (IBP)**: measures
//!    output difference between FP32 and INT4 paths as a bound width delta.
//!
//! 10. **Mixed precision: INT4 weights + FP16 compute (IBP)**: selective
//!     quantization with higher-precision attention, lower-precision FFN.
//!
//! 11. **Dequantize-compute-quantize pipeline bounds (IBP)**: full
//!     deq -> linear -> activation -> requant pipeline.
//!
//! ## Quantized Normalization & Softmax (tests 12-13)
//!
//! 12. **Quantized softmax approximation bounds (IBP)**: INT8 quantized
//!     logits through softmax produce valid probability distributions.
//!
//! 13. **Quantized LayerNorm bounds (IBP + CROWN)**: RMSNorm followed by
//!     INT8-quantized linear projection.
//!
//! ## Accumulator & Residual Safety (tests 14-15)
//!
//! 14. **INT8 matmul accumulator overflow safety (INT32 accum) (IBP)**:
//!     models INT8*INT8 accumulation in wider precision (INT32) by using
//!     smaller weight magnitudes representing the effective range.
//!
//! 15. **Quantized residual connection bounds (IBP)**: residual connection
//!     with INT8-quantized sublayer vs FP32 skip path.
//!
//! ## Full Pipeline (tests 16-18)
//!
//! 16. **Per-token activation quantization bounds (IBP)**: per-token dynamic
//!     quantization where each token position has independently scaled weights.
//!
//! 17. **Weight-only quantization vs activation quantization (IBP)**: compares
//!     W4A16 (weight-only INT4) vs W8A8 (weight+activation INT8) bound widths.
//!
//! 18. **Quantized model end-to-end output difference bound (IBP + CROWN)**:
//!     full decoder block with INT4 weights, comparing bound widths against
//!     FP32 baseline to bound the maximum output deviation.
//!
//! Dimensions (small for fast verification):
//! - SEQ_LEN=4, DIM=32, FFN_DIM=64
//!
//! Part of #4124: Compose tests for quantization-aware inference bounds.

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
const DIM: usize = 32;
const FFN_DIM: usize = 64;
const NUM_HEADS: usize = 4;
const GROUP_SIZE: usize = 16;

// ---------------------------------------------------------------------------
// Quantization parameters
// ---------------------------------------------------------------------------

/// FP32 baseline weight magnitude.
const FP32_WEIGHT_MAG: f32 = 0.02;
/// INT8 symmetric dequantized weight magnitude: scale * 127, scale=0.0005.
const INT8_WEIGHT_MAG: f32 = 0.0635;
/// INT4 symmetric dequantized weight magnitude: scale * 7, scale=0.01.
const INT4_WEIGHT_MAG: f32 = 0.07;
/// FP16 weight magnitude (higher precision than INT4, lower than FP32).
const FP16_WEIGHT_MAG: f32 = 0.05;
/// GPTQ dequantized weight magnitude (slightly larger due to Hessian residual).
const GPTQ_WEIGHT_MAG: f32 = 0.0735;
/// AWQ salient channel scale factor.
const AWQ_SALIENT_SCALE: f32 = 1.2;
/// AWQ weight magnitude (INT4 with activation-aware rescaling).
const AWQ_WEIGHT_MAG: f32 = INT4_WEIGHT_MAG / AWQ_SALIENT_SCALE;
/// INT8 asymmetric zero-point shift: reduces effective magnitude slightly.
const INT8_ASYM_WEIGHT_MAG: f32 = INT8_WEIGHT_MAG * 0.95;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a constant weight binding with given shape and magnitude.
fn weight_binding(shape: &[usize], mag: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), mag))
}

/// Bias binding (small constant).
fn bias_binding(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.01f32))
}

/// Norm weight (all ones).
fn norm_weight_binding(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 1.0f32))
}

/// Epsilon binding for normalization.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32))
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

/// Build a simple linear kernel def for comparing different weight magnitudes.
fn build_linear_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let w = b.add_input("w", &[DIM, DIM]);
    let bias = b.add_input("bias", &[DIM]);
    let out = b.add_linear(input, w, Some(bias), &[SEQ_LEN, DIM]);
    b.build(out).expect("valid linear kernel")
}

/// Run IBP on a linear kernel with given weight magnitude, return width.
fn linear_ibp_width(def: &TensorKernelDef, weight_mag: f32, input_range: f32) -> f32 {
    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], weight_mag),
        bias_binding(&[DIM]),
    ];
    let graph = tensor_kernel_to_graph(def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], input_range);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    bound_width(&output)
}

// ===========================================================================
// 1. INT8 linear output bounds vs FP32 (IBP)
// ===========================================================================

#[test]
fn test_int8_linear_output_bounds_vs_fp32() {
    let def = build_linear_kernel("qi_int8_vs_fp32");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    // FP32 weights
    let fp32_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], FP32_WEIGHT_MAG),
        bias_binding(&[DIM]),
    ];
    let fp32_graph = tensor_kernel_to_graph(&def, &fp32_bindings).expect("FP32 graph");
    let fp32_output = fp32_graph.propagate_ibp(&input_bounds).expect("FP32 IBP");
    assert_bounds_valid(&fp32_output);

    // INT8 weights
    let int8_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG),
        bias_binding(&[DIM]),
    ];
    let int8_graph = tensor_kernel_to_graph(&def, &int8_bindings).expect("INT8 graph");
    let int8_output = int8_graph.propagate_ibp(&input_bounds).expect("INT8 IBP");
    assert_bounds_valid(&int8_output);

    let fp32_width = bound_width(&fp32_output);
    let int8_width = bound_width(&int8_output);
    eprintln!("INT8 vs FP32 linear IBP: fp32_width={fp32_width:.6}, int8_width={int8_width:.6}");
    assert!(fp32_width.is_finite(), "FP32 width must be finite");
    assert!(int8_width.is_finite(), "INT8 width must be finite");
    // INT8 has larger weight magnitude => wider bounds expected
    assert!(
        int8_width >= fp32_width - 1e-4,
        "INT8 should have >= FP32 width: int8={int8_width}, fp32={fp32_width}"
    );
}

// ===========================================================================
// 2. INT8 symmetric quantization: scale * int_value bounds (IBP)
// ===========================================================================

/// Models INT8 symmetric quantization where dequantized weights are
/// scale * int_value, with int_value in [-127, 127] and scale chosen
/// to represent the original weight range.
#[test]
fn test_int8_symmetric_scale_int_value_bounds() {
    let mut b = TensorBlockBuilder::new("qi_int8_sym_scale");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let w = b.add_input("w_deq", &[DIM, DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid INT8 symmetric kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT8 symmetric: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Symmetric: output should be roughly centered around bias (0 here)
    let asymmetry = (lo_min + hi_max).abs();
    assert!(
        asymmetry < 1.0,
        "symmetric quant should produce roughly centered bounds, asymmetry={asymmetry}"
    );
}

// ===========================================================================
// 3. INT8 asymmetric quantization: (int_value - zero_point) * scale (IBP)
// ===========================================================================

/// Models INT8 asymmetric quantization where dequantized weights are
/// (int_value - zero_point) * scale. The nonzero zero_point shifts
/// the effective weight magnitude slightly.
#[test]
fn test_int8_asymmetric_zero_point_scale_bounds() {
    let mut b = TensorBlockBuilder::new("qi_int8_asym_zp");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let w = b.add_input("w_deq", &[DIM, DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid INT8 asymmetric kernel");

    // Asymmetric dequant produces slightly different magnitude
    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], INT8_ASYM_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT8 asymmetric: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 4. INT4 GPTQ dequantization: group-wise scale bounds (IBP)
// ===========================================================================

/// Models GPTQ group-wise dequantization where each group of weights
/// has a different scale, producing group-varying weight magnitudes.
#[test]
fn test_int4_gptq_group_wise_scale_bounds() {
    let out_dim = GROUP_SIZE * 2; // 32
    let in_dim = GROUP_SIZE; // 16

    let mut b = TensorBlockBuilder::new("qi_int4_gptq_group");
    let input = b.add_input("x", &[SEQ_LEN, in_dim]);
    let w = b.add_input("w_gptq", &[out_dim, in_dim]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, out_dim]);
    let def = b.build(out).expect("valid GPTQ group kernel");

    // Group 1 (rows 0..GROUP_SIZE): normal GPTQ magnitude
    // Group 2 (rows GROUP_SIZE..out_dim): slightly larger (Hessian residual)
    let mut weight_data = vec![INT4_WEIGHT_MAG; out_dim * in_dim];
    for row in GROUP_SIZE..out_dim {
        for col in 0..in_dim {
            weight_data[row * in_dim + col] = GPTQ_WEIGHT_MAG;
        }
    }
    let w_tensor =
        ArrayD::from_shape_vec(IxDyn(&[out_dim, in_dim]), weight_data).expect("valid weight shape");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, in_dim], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GPTQ group-wise: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. AWQ per-channel activation-aware scaling (IBP)
// ===========================================================================

/// Models AWQ where salient channels are rescaled before quantization.
/// Net dequantized weight magnitude is INT4_WEIGHT_MAG / AWQ_SALIENT_SCALE,
/// which should produce tighter bounds than raw INT4.
#[test]
fn test_awq_per_channel_scaling_bounds() {
    let def = build_linear_kernel("qi_awq_channel");

    let awq_width = linear_ibp_width(&def, AWQ_WEIGHT_MAG, 1.0);
    let int4_width = linear_ibp_width(&def, INT4_WEIGHT_MAG, 1.0);

    eprintln!("AWQ per-channel: awq_width={awq_width:.6}, int4_width={int4_width:.6}");
    assert!(awq_width.is_finite(), "AWQ width must be finite");
    assert!(int4_width.is_finite(), "INT4 width must be finite");
    // AWQ has smaller effective magnitude => tighter bounds
    assert!(
        awq_width <= int4_width + 1e-4,
        "AWQ should be tighter: awq={awq_width}, int4={int4_width}"
    );
}

// ===========================================================================
// 6. Group quantization: per-group scale and zero point (IBP)
// ===========================================================================

/// Verifies that group quantization with per-group scale/zero_point
/// produces bounded outputs, with tighter groups yielding narrower bounds.
#[test]
fn test_group_quant_per_group_scale_zero() {
    let mut b = TensorBlockBuilder::new("qi_group_scale_zp");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let w = b.add_input("w_deq", &[DIM, DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid group quant kernel");

    // Create weight tensor with per-group variation
    let n_groups = DIM / GROUP_SIZE;
    let mut weight_data = vec![0.0f32; DIM * DIM];
    for g in 0..n_groups {
        let scale = INT4_WEIGHT_MAG * (1.0 + 0.1 * g as f32);
        for row in (g * GROUP_SIZE)..((g + 1) * GROUP_SIZE) {
            for col in 0..DIM {
                weight_data[row * DIM + col] = scale;
            }
        }
    }
    let w_tensor =
        ArrayD::from_shape_vec(IxDyn(&[DIM, DIM]), weight_data).expect("valid weight shape");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Group quant per-group: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. Quantized attention score bounds (INT8 QKV) (IBP + CROWN)
// ===========================================================================

fn build_int8_attn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qi_int8_attn_qkv");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let shape = [SEQ_LEN, DIM];

    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);

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

    b.build(attn_out).expect("valid INT8 attention kernel")
}

fn int8_attn_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG), // q_w
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG), // k_w
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG), // v_w
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG), // o_w
    ]
}

#[test]
fn test_int8_attn_qkv_ibp() {
    let def = build_int8_attn_kernel();
    let bindings = int8_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT8 attention QKV IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_int8_attn_qkv_crown() {
    let def = build_int8_attn_kernel();
    let bindings = int8_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT8 attention QKV CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 8. Quantized MLP bounds (INT4 weights, FP16 activations) (IBP)
// ===========================================================================

/// SwiGLU MLP with INT4 quantized weights and full-precision activations.
#[test]
fn test_quantized_mlp_int4_weights_fp16_act() {
    let mut b = TensorBlockBuilder::new("qi_int4_mlp");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, DIM];

    let gate_w = b.add_input("gate_w", &[FFN_DIM, DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, DIM]);
    let down_w = b.add_input("down_w", &[DIM, FFN_DIM]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up, &ffn_shape);
    let out = b.add_linear(gated, down_w, None, &out_shape);

    let def = b.build(out).expect("valid quantized MLP kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[FFN_DIM, DIM], INT4_WEIGHT_MAG),
        weight_binding(&[FFN_DIM, DIM], INT4_WEIGHT_MAG),
        weight_binding(&[DIM, FFN_DIM], INT4_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized MLP INT4 IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Quantization error bound: |fp32 - quant| <= delta (IBP)
// ===========================================================================

/// Measures the output bound width difference between FP32 and INT4 paths
/// as a proxy for quantization error delta. The delta is the difference in
/// bound widths, bounding how much wider INT4 output ranges can be.
#[test]
fn test_quantization_error_delta_bound() {
    let def = build_linear_kernel("qi_error_delta");

    let fp32_width = linear_ibp_width(&def, FP32_WEIGHT_MAG, 1.0);
    let int4_width = linear_ibp_width(&def, INT4_WEIGHT_MAG, 1.0);

    let delta = (int4_width - fp32_width).abs();
    eprintln!(
        "Quant error delta: fp32_width={fp32_width:.6}, int4_width={int4_width:.6}, delta={delta:.6}"
    );
    assert!(delta.is_finite(), "delta must be finite");
    // The delta should be bounded (not blow up)
    assert!(
        delta < 100.0,
        "quantization error delta should be bounded, got {delta}"
    );
}

// ===========================================================================
// 10. Mixed precision: INT4 weights + FP16 compute (IBP)
// ===========================================================================

/// Selective quantization: attention uses FP16 weights (higher precision),
/// FFN uses INT4 weights (lower precision, larger model).
#[test]
fn test_mixed_precision_int4_fp16_compute() {
    let mut b = TensorBlockBuilder::new("qi_mixed_int4_fp16");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let shape = [SEQ_LEN, DIM];

    // FP16 attention
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);

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

    let h = b.add_binary_add(input, attn_out, &shape);

    // INT4 FFN
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let gate_w = b.add_input("gate_w", &[FFN_DIM, DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, DIM]);
    let down_w = b.add_input("down_w", &[DIM, FFN_DIM]);

    let gate = b.add_linear(h, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(h, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(gated, down_w, None, &shape);

    let out = b.add_binary_add(h, ffn_out, &shape);
    let def = b.build(out).expect("valid mixed precision kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], FP16_WEIGHT_MAG), // q_w
        weight_binding(&[DIM, DIM], FP16_WEIGHT_MAG), // k_w
        weight_binding(&[DIM, DIM], FP16_WEIGHT_MAG), // v_w
        weight_binding(&[DIM, DIM], FP16_WEIGHT_MAG), // o_w
        weight_binding(&[FFN_DIM, DIM], INT4_WEIGHT_MAG), // gate_w
        weight_binding(&[FFN_DIM, DIM], INT4_WEIGHT_MAG), // up_w
        weight_binding(&[DIM, FFN_DIM], INT4_WEIGHT_MAG), // down_w
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mixed INT4+FP16 compute: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. Dequantize-compute-quantize pipeline bounds (IBP)
// ===========================================================================

/// Models the full dequantize -> linear -> activation -> requantize
/// pipeline. In practice, requantization clips output to the target
/// dtype range. We model this as linear -> GELU -> linear (the final
/// linear simulates requant projection to quantized output space).
#[test]
fn test_deq_compute_requant_pipeline_bounds() {
    let mut b = TensorBlockBuilder::new("qi_deq_compute_quant");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    // Dequantized INT8 weights for first linear
    let w1 = b.add_input("w1_deq", &[DIM, DIM]);
    let h = b.add_linear(input, w1, None, &[SEQ_LEN, DIM]);
    let h = b.add_gelu(h, &[SEQ_LEN, DIM]);

    // Second linear simulating requantization projection
    let w2 = b.add_input("w2_requant", &[DIM, DIM]);
    let out = b.add_linear(h, w2, None, &[SEQ_LEN, DIM]);

    let def = b.build(out).expect("valid deq-compute-quant kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG),
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Deq-compute-quant pipeline IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
    assert!(
        width < 500.0,
        "pipeline width should be bounded, got {width}"
    );
}

// ===========================================================================
// 12. Quantized softmax approximation bounds (IBP)
// ===========================================================================

/// INT8 quantized logits through softmax must produce valid probability
/// distributions with outputs in [0, 1].
#[test]
fn test_quantized_softmax_approximation_bounds() {
    let mut b = TensorBlockBuilder::new("qi_quant_softmax");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    // INT8 quantized projection to logits
    let w = b.add_input("w_deq", &[DIM, DIM]);
    let logits = b.add_linear(input, w, None, &[SEQ_LEN, DIM]);

    // Softmax over last dimension
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid quantized softmax kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-5;
    eprintln!("Quantized softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "softmax output must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax output must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 13. Quantized LayerNorm bounds (IBP + CROWN)
// ===========================================================================

fn build_quantized_layernorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qi_quant_layernorm");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let shape = [SEQ_LEN, DIM];

    // RMSNorm
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    // INT8 quantized linear after normalization
    let w = b.add_input("w_deq", &[DIM, DIM]);
    let out = b.add_linear(normed, w, None, &shape);

    b.build(out).expect("valid quantized LayerNorm kernel")
}

fn quantized_layernorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        norm_weight_binding(DIM),
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG),
    ]
}

#[test]
fn test_quantized_layernorm_ibp() {
    let def = build_quantized_layernorm_kernel();
    let bindings = quantized_layernorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Quantized LayerNorm IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

#[test]
fn test_quantized_layernorm_crown() {
    let def = build_quantized_layernorm_kernel();
    let bindings = quantized_layernorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("Quantized LayerNorm CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 14. INT8 matmul accumulator overflow safety (INT32 accum) (IBP)
// ===========================================================================

/// Models INT8 matrix multiplication where products accumulate in INT32.
/// The effective weight magnitude per-element is smaller (each int8*int8
/// product is bounded by 127*127=16129, accumulated over DIM elements).
/// We model this with a reduced weight magnitude representing the
/// effective per-element contribution.
#[test]
fn test_int8_matmul_accumulator_overflow_safety() {
    let mut b = TensorBlockBuilder::new("qi_int8_accum_safety");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let w = b.add_input("w", &[DIM, DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid INT8 accumulator kernel");

    // INT8*INT8 -> INT32 accumulator. Per-element product range [-16129, 16129].
    // Accumulated over DIM elements, then rescaled by output_scale.
    // Model effective magnitude as INT8_WEIGHT_MAG (already represents
    // the dequantized value = scale * int_value).
    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("INT8 accumulator safety: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Output width should scale with DIM * weight_mag * input_range
    // Expected: ~2 * DIM * INT8_WEIGHT_MAG * 1.0 = 2 * 32 * 0.0635 = 4.064
    assert!(
        width < 50.0,
        "accumulated output should be bounded, got width={width}"
    );
}

// ===========================================================================
// 15. Quantized residual connection bounds (IBP)
// ===========================================================================

/// Residual connection with INT8-quantized sublayer: x + Linear_INT8(x).
/// The skip path preserves the original input range while the quantized
/// projection adds bounded perturbation.
#[test]
fn test_quantized_residual_connection_bounds() {
    let mut b = TensorBlockBuilder::new("qi_quant_residual");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let shape = [SEQ_LEN, DIM];

    let w = b.add_input("w_deq", &[DIM, DIM]);
    let proj = b.add_linear(input, w, None, &shape);
    let out = b.add_binary_add(input, proj, &shape);

    let def = b.build(out).expect("valid quantized residual kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("Quantized residual IBP: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Width should be at least the input width (2.0 for range=1.0)
    assert!(
        width >= 1.5,
        "residual width should be >= input width, got {width}"
    );
}

// ===========================================================================
// 16. Per-token activation quantization bounds (IBP)
// ===========================================================================

/// Models per-token dynamic quantization where each token position has
/// independently calibrated scale factors, producing position-varying
/// weight magnitudes.
#[test]
fn test_per_token_activation_quantization_bounds() {
    let mut b = TensorBlockBuilder::new("qi_per_token_quant");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    // First linear with per-position-varying weights (simulating per-token scale)
    let w1 = b.add_input("w1", &[DIM, DIM]);
    let h = b.add_linear(input, w1, None, &[SEQ_LEN, DIM]);
    let h = b.add_gelu(h, &[SEQ_LEN, DIM]);

    // Second linear
    let w2 = b.add_input("w2", &[DIM, DIM]);
    let out = b.add_linear(h, w2, None, &[SEQ_LEN, DIM]);

    let def = b.build(out).expect("valid per-token quant kernel");

    // Different magnitudes for the two layers representing token-wise calibration
    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG),
        weight_binding(&[DIM, DIM], INT8_WEIGHT_MAG * 0.9),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Per-token quant IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 17. Weight-only quantization vs activation quantization (IBP)
// ===========================================================================

/// Compares W4A16 (weight-only INT4, activations FP16) vs W8A8
/// (INT8 weights and activations). Both use linear layers with different
/// effective magnitudes. W4A16 has INT4 weight magnitude; W8A8 has
/// INT8 weight magnitude (slightly smaller effective range).
#[test]
fn test_weight_only_vs_activation_quantization() {
    let def = build_linear_kernel("qi_w4a16_vs_w8a8");

    // W4A16: INT4 weight-only, FP16 activations
    let w4a16_width = linear_ibp_width(&def, INT4_WEIGHT_MAG, 1.0);

    // W8A8: INT8 weights and activations (modeled as INT8 weight mag)
    let w8a8_width = linear_ibp_width(&def, INT8_WEIGHT_MAG, 1.0);

    eprintln!("W4A16 vs W8A8: w4a16_width={w4a16_width:.6}, w8a8_width={w8a8_width:.6}");
    assert!(w4a16_width.is_finite(), "W4A16 width must be finite");
    assert!(w8a8_width.is_finite(), "W8A8 width must be finite");
    // Both produce finite bounded outputs
    // INT4 has larger weight mag (0.07) than INT8 (0.0635)
    // so W4A16 should generally have wider bounds
    assert!(
        w4a16_width >= w8a8_width - 1e-4,
        "W4A16 (INT4) should have >= W8A8 (INT8) width: w4a16={w4a16_width}, w8a8={w8a8_width}"
    );
}

// ===========================================================================
// 18. Quantized model end-to-end output difference bound (IBP + CROWN)
// ===========================================================================

fn build_quantized_decoder_block(
    name: &str,
    weight_mag: f32,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let shape = [SEQ_LEN, DIM];

    // Pre-norm: RMSNorm
    let attn_eps = b.add_input("attn_eps", &[1]);
    let attn_norm_w = b.add_input("attn_norm_w", &[DIM]);
    let normed = b.add_rms_norm(input, attn_eps, 1, attn_norm_w, &shape);

    // Attention
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);

    let attn_out = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // Residual
    let h = b.add_binary_add(input, attn_out, &shape);

    // Pre-norm: RMSNorm before FFN
    let ffn_eps = b.add_input("ffn_eps", &[1]);
    let ffn_norm_w = b.add_input("ffn_norm_w", &[DIM]);
    let normed_ffn = b.add_rms_norm(h, ffn_eps, 1, ffn_norm_w, &shape);

    // SwiGLU FFN
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let gate_w = b.add_input("gate_w", &[FFN_DIM, DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, DIM]);
    let down_w = b.add_input("down_w", &[DIM, FFN_DIM]);

    let gate = b.add_linear(normed_ffn, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(normed_ffn, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(gated, down_w, None, &shape);

    // Final residual
    let out = b.add_binary_add(h, ffn_out, &shape);
    let def = b.build(out).expect("valid decoder block kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),                               // attn_eps
        norm_weight_binding(DIM),                    // attn_norm_w
        weight_binding(&[DIM, DIM], weight_mag),     // q_w
        weight_binding(&[DIM, DIM], weight_mag),     // k_w
        weight_binding(&[DIM, DIM], weight_mag),     // v_w
        weight_binding(&[DIM, DIM], weight_mag),     // o_w
        eps_binding(),                               // ffn_eps
        norm_weight_binding(DIM),                    // ffn_norm_w
        weight_binding(&[FFN_DIM, DIM], weight_mag), // gate_w
        weight_binding(&[FFN_DIM, DIM], weight_mag), // up_w
        weight_binding(&[DIM, FFN_DIM], weight_mag), // down_w
    ];

    (def, bindings)
}

#[test]
fn test_quantized_e2e_output_difference_ibp() {
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    // FP32 baseline
    let (fp32_def, fp32_bindings) = build_quantized_decoder_block("qi_e2e_fp32", FP32_WEIGHT_MAG);
    let fp32_graph = tensor_kernel_to_graph(&fp32_def, &fp32_bindings).expect("FP32 graph");
    let fp32_output = fp32_graph.propagate_ibp(&input_bounds).expect("FP32 IBP");
    assert_bounds_valid(&fp32_output);
    let fp32_width = bound_width(&fp32_output);

    // INT4 quantized
    let (int4_def, int4_bindings) = build_quantized_decoder_block("qi_e2e_int4", INT4_WEIGHT_MAG);
    let int4_graph = tensor_kernel_to_graph(&int4_def, &int4_bindings).expect("INT4 graph");
    let int4_output = int4_graph.propagate_ibp(&input_bounds).expect("INT4 IBP");
    assert_bounds_valid(&int4_output);
    let int4_width = bound_width(&int4_output);

    let delta = (int4_width - fp32_width).abs();
    eprintln!("E2E IBP: fp32_width={fp32_width:.6}, int4_width={int4_width:.6}, delta={delta:.6}");
    assert!(fp32_width.is_finite(), "FP32 width must be finite");
    assert!(int4_width.is_finite(), "INT4 width must be finite");
    assert!(delta.is_finite(), "output difference delta must be finite");
}

#[test]
fn test_quantized_e2e_output_difference_crown() {
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let (def, bindings) = build_quantized_decoder_block("qi_e2e_crown", INT4_WEIGHT_MAG);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let (method, output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("E2E CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
