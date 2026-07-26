// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for quantized weight inference (INT4 GPTQ/AWQ) bound
//! propagation through full model inference paths.
//!
//! Verifies NY IBP and CROWN bound propagation through inference
//! pipelines using INT4-quantized weights. Unlike `compose_dpdf_quantized.rs`
//! (dequantization mechanics) and `compose_dpdf_quantization.rs` (GPTQ/AWQ
//! dequant variants + precision analysis), this file covers the inference-time
//! bound behaviour of complete model sub-blocks with quantized weights: how
//! INT4 dequantized weight bounds interact with activations, residuals,
//! normalization, attention, and output heads during inference.
//!
//! ## INT4 Dequantization Fundamentals (tests 1-3)
//!
//! 1. INT4 dequantization: scale * (q - zero_point) bounds (IBP)
//! 2. Group quantization: per-group scale/zero bounds (IBP)
//! 3. Quantized linear layer output bounds (IBP + CROWN)
//!
//! ## Precision & Scheme Comparison (tests 4-6)
//!
//! 4. INT4 vs FP16 output bound comparison (IBP)
//! 5. GPTQ quantization error bounds (IBP)
//! 6. AWQ activation-aware quantization bounds (IBP)
//!
//! ## Quantized Inference Sub-Blocks (tests 7-11)
//!
//! 7. Quantized attention QKV projection bounds (IBP + CROWN)
//! 8. Quantized FFN (SwiGLU with INT4 weights) bounds (IBP)
//! 9. Mixed precision: INT4 weights + FP16 activations (IBP)
//! 10. Quantized residual connection bounds (IBP)
//! 11. Quantized LayerNorm interaction bounds (IBP + CROWN)
//!
//! ## Advanced Quantized Inference (tests 12-15)
//!
//! 12. INT8 vs INT4 quantization precision comparison (IBP)
//! 13. Quantized embedding lookup bounds (IBP)
//! 14. Quantized detection head output bounds (IBP)
//! 15. Full quantized decoder block: attention + FFN + residual (IBP + CROWN)
//!
//! INT4 weight-only quantization (W4A16) scheme:
//!   Weights dequantized at inference time: w_f32 = (code - zero_point) * scale
//!   Activations remain in FP16/FP32 (full precision).
//!   GPTQ: Hessian-based per-group quantization minimizing reconstruction error.
//!   AWQ: Activation-aware scaling of salient channels before quantization.
//!
//! Architecture references:
//! - GPTQ (Frantar et al., 2022): Post-training quantization via approximate second-order
//! - AWQ (Lin et al., 2023): Activation-aware weight quantization for LLMs
//! - Qwen3-VL: SwiGLU + GQA decoder with INT4 GPTQ deployment
//! - Granite-Docling: INT4 quantized Granite LLM decoder for document understanding
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, FFN_DIM=128
//!
//! Part of #4026: Quantized weight inference compose tests.

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
const VOCAB_SIZE: usize = 256;
const EMBED_DIM: usize = 64;
const GROUP_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Quantization parameters
// ---------------------------------------------------------------------------

/// FP32 baseline weight magnitude.
const FP32_WEIGHT_MAG: f32 = 0.02;
/// INT4 symmetric dequantized weight magnitude: scale * 7, scale=0.01.
const INT4_WEIGHT_MAG: f32 = 0.07;
/// INT8 symmetric dequantized weight magnitude: scale * 127, scale=0.0005.
const INT8_WEIGHT_MAG: f32 = 0.0635;
/// FP16 weight magnitude (higher precision than INT4, lower than FP32).
const FP16_WEIGHT_MAG: f32 = 0.05;
/// GPTQ dequantized weight magnitude (slightly larger due to Hessian residual).
const GPTQ_WEIGHT_MAG: f32 = 0.0735;
/// AWQ salient channel scale factor.
const AWQ_SALIENT_SCALE: f32 = 1.2;
/// AWQ weight magnitude (INT4 with activation-aware rescaling).
const AWQ_WEIGHT_MAG: f32 = INT4_WEIGHT_MAG / AWQ_SALIENT_SCALE;

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

/// Build a standard SwiGLU FFN block with given weight magnitude.
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
// 1. INT4 dequantization: scale * (q - zero_point) bounds (IBP)
// ===========================================================================

/// Verify that INT4 dequantized weight -> linear produces finite, valid bounds.
/// Models dequantization as: w_deq = scale * (code - zero_point).
/// Since TensorBlockBuilder lacks a dequant op, we model the dequantized
/// weight range via constant tensors bounded by INT4_WEIGHT_MAG.
#[test]
fn test_int4_dequant_linear_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_int4_dequant");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w_deq", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);
    let out = b.add_linear(input, w, Some(bias), &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid INT4 dequant linear kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM], 0.01),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT4 dequant linear IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Group quantization: per-group scale/zero bounds (IBP)
// ===========================================================================

/// Verify group-quantized linear where weight groups have different magnitudes.
/// Models GROUP_SIZE-wise dequantization: each group of input channels shares
/// one scale/zero_point, producing group-varying weight ranges.
#[test]
fn test_group_quant_per_group_bounds_ibp() {
    // Model: two groups with different magnitudes (simulating per-group scale)
    let out_dim = GROUP_SIZE * 2; // 64
    let in_dim = GROUP_SIZE; // 32

    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_group_bounds");
    let input = b.add_input("x", &[SEQ_LEN, in_dim]);
    let w = b.add_input("w_deq", &[out_dim, in_dim]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, out_dim]);
    let def = b.build(out).expect("valid group quant kernel");

    // Group 1 has tighter range than group 2 (simulating different scales)
    let mut weight_data = vec![INT4_WEIGHT_MAG * 0.5; out_dim * in_dim];
    for row in GROUP_SIZE..out_dim {
        for col in 0..in_dim {
            weight_data[row * in_dim + col] = INT4_WEIGHT_MAG;
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
    eprintln!("Group quant per-group IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. Quantized linear layer output bounds (IBP + CROWN)
// ===========================================================================

fn build_quantized_linear_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_linear");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w_deq", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);
    let out = b.add_linear(input, w, Some(bias), &[SEQ_LEN, HIDDEN_DIM]);
    b.build(out).expect("valid quantized linear kernel")
}

fn quantized_linear_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM], 0.01),
    ]
}

#[test]
fn test_quantized_linear_ibp() {
    let def = build_quantized_linear_kernel();
    let bindings = quantized_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Quantized linear IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

#[test]
fn test_quantized_linear_crown() {
    let def = build_quantized_linear_kernel();
    let bindings = quantized_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("Quantized linear CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 4. INT4 vs FP16 output bound comparison (IBP)
// ===========================================================================

/// Compare output bound widths of INT4-quantized vs FP16-precision linear.
/// INT4 has smaller effective weight range, which should produce tighter or
/// comparable bounds (depending on weight magnitude calibration).
#[test]
fn test_int4_vs_fp16_bound_comparison_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_int4_fp16");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid linear kernel");

    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // INT4 weights
    let int4_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG),
    ];
    let int4_graph = tensor_kernel_to_graph(&def, &int4_bindings).expect("INT4 graph");
    let int4_output = int4_graph.propagate_ibp(&input_bounds).expect("INT4 IBP");
    assert_bounds_valid(&int4_output);
    let int4_width = bound_width(&int4_output);

    // FP16 weights (larger magnitude)
    let fp16_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP16_WEIGHT_MAG),
    ];
    let fp16_graph = tensor_kernel_to_graph(&def, &fp16_bindings).expect("FP16 graph");
    let fp16_output = fp16_graph.propagate_ibp(&input_bounds).expect("FP16 IBP");
    assert_bounds_valid(&fp16_output);
    let fp16_width = bound_width(&fp16_output);

    eprintln!("INT4 vs FP16 IBP: int4_width={int4_width:.6}, fp16_width={fp16_width:.6}");
    // Both should produce finite, reasonable bounds
    assert!(int4_width.is_finite(), "INT4 width must be finite");
    assert!(fp16_width.is_finite(), "FP16 width must be finite");
}

// ===========================================================================
// 5. GPTQ quantization error bounds (IBP)
// ===========================================================================

/// GPTQ dequantized weights have slightly larger magnitude than naive INT4
/// due to Hessian-based reordering residual. Verify that the larger magnitude
/// propagates to wider but still finite bounds.
#[test]
fn test_gptq_quantization_error_bounds_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_gptq_error");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w_gptq", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid GPTQ kernel");

    // GPTQ weights: slightly larger than raw INT4 symmetric
    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], GPTQ_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GPTQ graph");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("GPTQ IBP");
    assert_bounds_valid(&output);

    // Compare with raw INT4
    let int4_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG),
    ];
    let int4_graph = tensor_kernel_to_graph(&def, &int4_bindings).expect("INT4 graph");
    let int4_output = int4_graph.propagate_ibp(&input_bounds).expect("INT4 IBP");

    let gptq_width = bound_width(&output);
    let int4_width = bound_width(&int4_output);
    eprintln!("GPTQ error bounds IBP: gptq_width={gptq_width:.6}, int4_width={int4_width:.6}");
    // GPTQ has larger weight magnitude => wider bounds
    assert!(
        gptq_width >= int4_width - 1e-4,
        "GPTQ bounds should be >= INT4: gptq={gptq_width}, int4={int4_width}"
    );
}

// ===========================================================================
// 6. AWQ activation-aware quantization bounds (IBP)
// ===========================================================================

/// AWQ rescales salient channels before quantization, then divides back after
/// dequantization. Net weight magnitude is INT4_WEIGHT_MAG / AWQ_SALIENT_SCALE.
/// Verify this produces tighter bounds than raw INT4.
#[test]
fn test_awq_activation_aware_bounds_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_awq");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w_awq", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid AWQ kernel");

    // AWQ weights: smaller effective magnitude after activation-aware rescaling
    let awq_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], AWQ_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &awq_bindings).expect("AWQ graph");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let awq_output = graph.propagate_ibp(&input_bounds).expect("AWQ IBP");
    assert_bounds_valid(&awq_output);

    // Compare with raw INT4
    let int4_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG),
    ];
    let int4_graph = tensor_kernel_to_graph(&def, &int4_bindings).expect("INT4 graph");
    let int4_output = int4_graph.propagate_ibp(&input_bounds).expect("INT4 IBP");

    let awq_width = bound_width(&awq_output);
    let int4_width = bound_width(&int4_output);
    eprintln!("AWQ bounds IBP: awq_width={awq_width:.6}, int4_width={int4_width:.6}");
    // AWQ has smaller effective weight magnitude => tighter bounds
    assert!(
        awq_width <= int4_width + 1e-4,
        "AWQ bounds should be tighter: awq={awq_width}, int4={int4_width}"
    );
}

// ===========================================================================
// 7. Quantized attention QKV projection bounds (IBP + CROWN)
// ===========================================================================

fn build_quantized_attn_qkv_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_attn_qkv");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Q, K, V projections with INT4 quantized weights
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

    b.build(attn_out).expect("valid quantized attention kernel")
}

fn quantized_attn_qkv_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG), // q_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG), // k_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG), // v_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG), // o_w
    ]
}

#[test]
fn test_quantized_attn_qkv_ibp() {
    let def = build_quantized_attn_qkv_kernel();
    let bindings = quantized_attn_qkv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized attention QKV IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_quantized_attn_qkv_crown() {
    let def = build_quantized_attn_qkv_kernel();
    let bindings = quantized_attn_qkv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Quantized attention QKV CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 8. Quantized FFN (SwiGLU with INT4 weights) bounds (IBP)
// ===========================================================================

/// SwiGLU FFN with all three projections (gate, up, down) using INT4 weights.
#[test]
fn test_quantized_swiglu_ffn_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_swiglu_ffn");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_swiglu_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let def = b.build(out).expect("valid quantized SwiGLU kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, INT4_WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized SwiGLU FFN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Mixed precision: INT4 weights + FP16 activations (IBP)
// ===========================================================================

/// Mixed precision inference: INT4 weights for FFN, FP16-magnitude weights
/// for attention (simulating selective quantization where attention stays
/// higher precision for accuracy).
#[test]
fn test_mixed_precision_int4_fp16_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_mixed_prec");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Attention with FP16-precision weights
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

    // First residual
    let h = b.add_binary_add(input, attn_out, &shape);

    // SwiGLU FFN with INT4 quantized weights
    let ffn_out = build_swiglu_block(&mut b, h, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Second residual
    let out = b.add_binary_add(h, ffn_out, &shape);
    let def = b.build(out).expect("valid mixed precision kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP16_WEIGHT_MAG), // q_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP16_WEIGHT_MAG), // k_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP16_WEIGHT_MAG), // v_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP16_WEIGHT_MAG), // o_w
    ];
    // FFN with INT4 quantized weights
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, INT4_WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mixed precision INT4+FP16 IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. Quantized residual connection bounds (IBP)
// ===========================================================================

/// Residual: x + Linear_INT4(x). The skip connection preserves the original
/// input range while adding the quantized projection output.
#[test]
fn test_quantized_residual_connection_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_residual");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // INT4 quantized linear projection
    let w = b.add_input("w_deq", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj = b.add_linear(input, w, None, &shape);

    // Residual: x + Linear_INT4(x)
    let out = b.add_binary_add(input, proj, &shape);
    let def = b.build(out).expect("valid quantized residual kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized residual IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Residual: input in [-1,1] + INT4 projection output -> bounded
    assert!(
        lo_min > -100.0,
        "residual lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 11. Quantized LayerNorm interaction bounds (IBP + CROWN)
// ===========================================================================

fn build_quantized_layernorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_layernorm");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // RMSNorm before quantized linear
    let eps = b.add_input("eps", &[1]);
    let norm_weight = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_weight, &shape);

    // INT4 quantized linear after normalization
    let w = b.add_input("w_deq", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(normed, w, None, &shape);

    b.build(out).expect("valid quantized LayerNorm kernel")
}

fn quantized_layernorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG),
    ]
}

#[test]
fn test_quantized_layernorm_ibp() {
    let def = build_quantized_layernorm_kernel();
    let bindings = quantized_layernorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

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
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("Quantized LayerNorm CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 12. INT8 vs INT4 quantization precision comparison (IBP)
// ===========================================================================

/// Compare INT8 and INT4 output bound widths through the same linear layer.
/// INT8 has 256 levels vs INT4's 16 levels, but similar magnitude range,
/// so effective weight coverage differs.
#[test]
fn test_int8_vs_int4_precision_comparison_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_int8_int4");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid precision comparison kernel");

    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // INT4 weights
    let int4_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG),
    ];
    let int4_graph = tensor_kernel_to_graph(&def, &int4_bindings).expect("INT4 graph");
    let int4_output = int4_graph.propagate_ibp(&input_bounds).expect("INT4 IBP");
    assert_bounds_valid(&int4_output);

    // INT8 weights (similar overall magnitude range)
    let int8_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT8_WEIGHT_MAG),
    ];
    let int8_graph = tensor_kernel_to_graph(&def, &int8_bindings).expect("INT8 graph");
    let int8_output = int8_graph.propagate_ibp(&input_bounds).expect("INT8 IBP");
    assert_bounds_valid(&int8_output);

    let int4_width = bound_width(&int4_output);
    let int8_width = bound_width(&int8_output);
    eprintln!("INT8 vs INT4 precision IBP: int4_width={int4_width:.6}, int8_width={int8_width:.6}");
    // Both produce finite bounds
    assert!(int4_width.is_finite(), "INT4 width must be finite");
    assert!(int8_width.is_finite(), "INT8 width must be finite");
}

// ===========================================================================
// 13. Quantized embedding lookup bounds (IBP)
// ===========================================================================

/// Embedding lookup with INT4-quantized embedding table. The embedding
/// table entries have INT4-range magnitudes, producing bounded outputs.
#[test]
fn test_quantized_embedding_lookup_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_embedding");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    // Model embedding as: Linear(input, embed_weight) where embed_weight
    // represents the quantized embedding table projected to hidden dim.
    let embed_w = b.add_input("embed_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let embedded = b.add_linear(input, embed_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Follow with a quantized projection layer (typical post-embedding path)
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(embedded, proj_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid quantized embedding kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG), // embed_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG), // proj_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized embedding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 14. Quantized detection head output bounds (IBP)
// ===========================================================================

/// Detection head with INT4-quantized linear -> sigmoid. The sigmoid output
/// must be bounded in (0, 1) regardless of quantization scheme.
#[test]
fn test_quantized_detection_head_ibp() {
    let num_classes = 16;
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_detect_head");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    // INT4 quantized classification head
    let cls_w = b.add_input("cls_w", &[num_classes, HIDDEN_DIM]);
    let cls_bias = b.add_input("cls_bias", &[num_classes]);
    let logits = b.add_linear(input, cls_w, Some(cls_bias), &[SEQ_LEN, num_classes]);

    // Sigmoid output -> bounded in (0, 1)
    let out = b.add_sigmoid(logits, &[SEQ_LEN, num_classes]);
    let def = b.build(out).expect("valid quantized detection head kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[num_classes, HIDDEN_DIM], INT4_WEIGHT_MAG),
        weight_binding(&[num_classes], 0.01),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Quantized detection head IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "sigmoid output lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "sigmoid output upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 15. Full quantized decoder block: attention + FFN + residual (IBP + CROWN)
// ===========================================================================

fn build_full_quantized_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_quant_inf_full_decoder");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Pre-norm: RMSNorm before attention
    let attn_eps = b.add_input("attn_eps", &[1]);
    let attn_norm_w = b.add_input("attn_norm_w", &[HIDDEN_DIM]);
    let normed_attn = b.add_rms_norm(input, attn_eps, 1, attn_norm_w, &shape);

    // INT4 quantized attention projections
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

    // First residual: x + Attention(RMSNorm(x))
    let h = b.add_binary_add(input, attn_out, &shape);

    // Pre-norm: RMSNorm before FFN
    let ffn_eps = b.add_input("ffn_eps", &[1]);
    let ffn_norm_w = b.add_input("ffn_norm_w", &[HIDDEN_DIM]);
    let normed_ffn = b.add_rms_norm(h, ffn_eps, 1, ffn_norm_w, &shape);

    // INT4 quantized SwiGLU FFN
    let ffn_out = build_swiglu_block(&mut b, normed_ffn, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Second residual: h + SwiGLU(RMSNorm(h))
    let out = b.add_binary_add(h, ffn_out, &shape);
    b.build(out).expect("valid full quantized decoder kernel")
}

fn full_quantized_decoder_bindings() -> Vec<TensorParamBinding> {
    let eps = TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32));
    let norm_w =
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32));

    let mut bindings = vec![
        TensorParamBinding::Variable,
        eps.clone(),                                                // attn_eps
        norm_w.clone(),                                             // attn_norm_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG), // q_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG), // k_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG), // v_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG), // o_w
        eps,                                                        // ffn_eps
        norm_w,                                                     // ffn_norm_w
    ];
    // INT4 quantized SwiGLU weights
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, INT4_WEIGHT_MAG);
    bindings
}

#[test]
fn test_full_quantized_decoder_ibp() {
    let def = build_full_quantized_decoder_kernel();
    let bindings = full_quantized_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full quantized decoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_full_quantized_decoder_crown() {
    let def = build_full_quantized_decoder_kernel();
    let bindings = full_quantized_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full quantized decoder CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
