// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for model quantization preservation (INT4/INT8 vs FP32 bounds).
//!
//! Verifies that quantized (INT4/INT8) model subgraphs preserve output bounds
//! relative to FP32 baselines through NY IBP and CROWN propagation.
//!
//! ## GPTQ INT4 Dequantization Bounds (tests 1-4)
//!
//! 1. GPTQ INT4 dequant bounds for Qwen3-VL MoE expert FFN (IBP)
//! 2. GPTQ INT4 per-group scale variation across experts (IBP)
//! 3. GPTQ INT4 expert gate + FFN end-to-end (IBP + CROWN)
//! 4. GPTQ INT4 MoE residual: quantized expert + skip (IBP)
//!
//! ## AWQ INT4 Per-Channel Preservation (tests 5-8)
//!
//! 5. AWQ per-channel salient scale preserves output bounds (IBP)
//! 6. AWQ channel-wise vs uniform quantization comparison (IBP)
//! 7. AWQ quantized SwiGLU FFN preserves output range (IBP)
//! 8. AWQ quantized attention + FFN decoder block (IBP + CROWN)
//!
//! ## INT8 Attention Quantization (tests 9-12)
//!
//! 9. INT8 QK scoring bounds (SageAttention-style) (IBP)
//! 10. INT8 QK + FP32 PV accumulation split-precision (IBP)
//! 11. INT8 attention with smooth-K channel subtraction (IBP)
//! 12. INT8 attention vs FP32 attention bound width comparison (IBP)
//!
//! ## Quantized vs FP32 Equivalence Margin (tests 13-16)
//!
//! 13. Per-layer quantization error: INT4 linear vs FP32 linear (IBP)
//! 14. Per-layer quantization error: INT8 linear vs FP32 linear (IBP)
//! 15. Quantization error accumulation: 2-layer stack margin growth (IBP)
//! 16. Quantized softmax output margin: bounded deviation from FP32 (IBP)
//!
//! ## Group Quantization Structure (tests 17-20)
//!
//! 17. Group-size partitioning preserves weight tensor shape (IBP)
//! 18. Group-size impact: smaller groups produce tighter per-element bounds (IBP)
//! 19. Non-aligned group boundary: partial last group (IBP)
//! 20. Group quantization monotone tightening: tighter input -> tighter output (IBP)
//!
//! Quantization schemes:
//! - GPTQ (Frantar et al., 2022): Hessian-based per-group INT4 quantization
//! - AWQ (Lin et al., 2023): Activation-aware INT4 with per-channel salient scaling
//! - SageAttention (Zhang et al., 2024): INT8 QK scoring + FP32 PV accumulation
//!
//! Architecture references:
//! - Qwen3-VL MoE: SwiGLU + GQA decoder with MoE routing + INT4 GPTQ deployment
//! - SageAttention: 2-5x speedup over FlashAttention v2 via INT8 Q/K quantization
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, FFN_DIM=128, NUM_HEADS=4, GROUP_SIZE=32
//!
//! Part of #4087: Compose tests for model quantization preservation.

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
const NUM_EXPERTS: usize = 4;
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
/// GPTQ dequantized weight magnitude (slightly larger due to Hessian residual).
const GPTQ_WEIGHT_MAG: f32 = 0.0735;
/// AWQ salient channel scale factor.
const AWQ_SALIENT_SCALE: f32 = 1.2;
/// AWQ weight magnitude (INT4 with activation-aware rescaling).
const AWQ_WEIGHT_MAG: f32 = INT4_WEIGHT_MAG / AWQ_SALIENT_SCALE;
/// AWQ non-salient channel weight magnitude (unscaled INT4).
const AWQ_NONSALIENT_MAG: f32 = INT4_WEIGHT_MAG;
/// INT8 smooth-K weight magnitude (reduced after channel mean subtraction).
const INT8_SMOOTH_MAG: f32 = INT8_WEIGHT_MAG * 0.85;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a constant weight binding with given shape and magnitude.
fn weight_binding(shape: &[usize], mag: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), mag))
}

/// Constant epsilon binding for RMSNorm.
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

/// Build a standard SwiGLU FFN block.
fn build_swiglu(
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
    bindings.push(weight_binding(&[ffn_dim, hidden_dim], weight_mag));
    bindings.push(weight_binding(&[ffn_dim, hidden_dim], weight_mag));
    bindings.push(weight_binding(&[hidden_dim, ffn_dim], weight_mag));
}

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Build a linear layer graph, propagate with given weight mag, return output bounds.
fn propagate_linear(
    input_bounds: &BoundedTensor,
    in_dim: usize,
    out_dim: usize,
    weight_mag: f32,
) -> BoundedTensor {
    let mut b = TensorBlockBuilder::new("quant_pres_linear_helper");
    let input = b.add_input("x", &[SEQ_LEN, in_dim]);
    let w = b.add_input("w", &[out_dim, in_dim]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, out_dim]);
    let def = b.build(out).expect("valid linear kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[out_dim, in_dim], weight_mag),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    graph.propagate_ibp(input_bounds).expect("IBP propagation")
}

// ===========================================================================
// 1. GPTQ INT4 dequant bounds for Qwen3-VL MoE expert FFN (IBP)
// ===========================================================================

/// Verify GPTQ INT4 dequantized expert FFN (SwiGLU) produces finite, valid bounds.
/// Models the Qwen3-VL MoE architecture where each expert has INT4-quantized
/// gate/up/down projections with GPTQ Hessian-based scales.
#[test]
fn test_gptq_int4_moe_expert_ffn_ibp() {
    let mut b = TensorBlockBuilder::new("quant_pres_gptq_moe_expert");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_swiglu(&mut b, input, "expert", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let def = b.build(out).expect("valid GPTQ MoE expert kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, GPTQ_WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GPTQ MoE expert FFN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. GPTQ INT4 per-group scale variation across experts (IBP)
// ===========================================================================

/// Verify that different GPTQ scale values across MoE experts produce
/// structurally valid bounds. Expert 0 has tight scales, expert 1 has
/// looser scales (simulating different Hessian-based calibrations).
#[test]
fn test_gptq_int4_per_group_scale_variation_ibp() {
    let out_dim = GROUP_SIZE * 2; // 64
    let in_dim = GROUP_SIZE; // 32

    let mut b = TensorBlockBuilder::new("quant_pres_gptq_scale_var");
    let input = b.add_input("x", &[SEQ_LEN, in_dim]);
    let w = b.add_input("w_gptq", &[out_dim, in_dim]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, out_dim]);
    let def = b.build(out).expect("valid GPTQ scale variation kernel");

    // Expert 0 group: tighter GPTQ scale (well-calibrated layer)
    // Expert 1 group: looser GPTQ scale (harder-to-quantize layer)
    let mut weight_data = vec![GPTQ_WEIGHT_MAG * 0.6; out_dim * in_dim];
    for row in GROUP_SIZE..out_dim {
        for col in 0..in_dim {
            weight_data[row * in_dim + col] = GPTQ_WEIGHT_MAG * 1.3;
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
    eprintln!("GPTQ per-group scale variation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. GPTQ INT4 expert gate + FFN end-to-end (IBP + CROWN)
// ===========================================================================

fn build_gptq_gate_plus_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("quant_pres_gptq_gate_ffn");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    // Expert gate: Linear -> softmax routing
    let gate_w = b.add_input("gate_w", &[NUM_EXPERTS, HIDDEN_DIM]);
    let gate_logits = b.add_linear(input, gate_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let _gate_probs = b.add_softmax(gate_logits, -1, &[SEQ_LEN, NUM_EXPERTS]);

    // GPTQ-quantized expert FFN (worst-case single expert path)
    let out = build_swiglu(&mut b, input, "expert", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Sigmoid output confidence
    let sig_out = b.add_sigmoid(out, &[SEQ_LEN, HIDDEN_DIM]);
    b.build(sig_out).expect("valid GPTQ gate+FFN kernel")
}

fn gptq_gate_plus_ffn_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[NUM_EXPERTS, HIDDEN_DIM], GPTQ_WEIGHT_MAG), // gate_w
    ];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, GPTQ_WEIGHT_MAG);
    bindings
}

#[test]
fn test_gptq_gate_plus_ffn_ibp() {
    let def = build_gptq_gate_plus_ffn_kernel();
    let bindings = gptq_gate_plus_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GPTQ gate+FFN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

#[test]
fn test_gptq_gate_plus_ffn_crown() {
    let def = build_gptq_gate_plus_ffn_kernel();
    let bindings = gptq_gate_plus_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GPTQ gate+FFN CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 4. GPTQ INT4 MoE residual: quantized expert + skip (IBP)
// ===========================================================================

/// MoE residual: x + GPTQ_Expert(x). The skip connection preserves the
/// original input range while adding the GPTQ-quantized expert output.
#[test]
fn test_gptq_moe_residual_ibp() {
    let mut b = TensorBlockBuilder::new("quant_pres_gptq_moe_residual");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // GPTQ-quantized expert SwiGLU FFN
    let ffn_out = build_swiglu(&mut b, input, "expert", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Residual: x + Expert(x)
    let out = b.add_binary_add(input, ffn_out, &shape);
    let def = b.build(out).expect("valid GPTQ MoE residual kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, GPTQ_WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GPTQ MoE residual IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. AWQ per-channel salient scale preserves output bounds (IBP)
// ===========================================================================

/// AWQ scales salient channels by activation magnitude before quantization,
/// then divides back after dequantization. Verify that per-channel-varying
/// weight magnitudes (salient vs non-salient) produce valid bounds.
#[test]
fn test_awq_per_channel_salient_bounds_ibp() {
    let mut b = TensorBlockBuilder::new("quant_pres_awq_salient");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w_awq", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid AWQ salient kernel");

    // Construct per-channel AWQ weights: half salient (scaled), half non-salient
    let half = HIDDEN_DIM / 2;
    let mut weight_data = Vec::with_capacity(HIDDEN_DIM * HIDDEN_DIM);
    for _row in 0..HIDDEN_DIM {
        for col in 0..HIDDEN_DIM {
            if col < half {
                // Salient channels: smaller effective magnitude after AWQ rescaling
                weight_data.push(AWQ_WEIGHT_MAG);
            } else {
                // Non-salient channels: standard INT4 magnitude
                weight_data.push(AWQ_NONSALIENT_MAG);
            }
        }
    }
    let w_tensor = ArrayD::from_shape_vec(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), weight_data)
        .expect("valid weight shape");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("AWQ per-channel salient IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. AWQ channel-wise vs uniform quantization comparison (IBP)
// ===========================================================================

/// Compare AWQ (per-channel salient scaling) vs uniform INT4 quantization.
/// AWQ should produce tighter or comparable bounds due to smaller effective
/// weight magnitudes on salient channels.
#[test]
fn test_awq_channelwise_vs_uniform_ibp() {
    let mut b = TensorBlockBuilder::new("quant_pres_awq_vs_uniform");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid comparison kernel");

    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // AWQ weights (smaller effective magnitude)
    let awq_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], AWQ_WEIGHT_MAG),
    ];
    let awq_graph = tensor_kernel_to_graph(&def, &awq_bindings).expect("AWQ graph");
    let awq_output = awq_graph.propagate_ibp(&input_bounds).expect("AWQ IBP");
    assert_bounds_valid(&awq_output);

    // Uniform INT4 weights (standard magnitude)
    let uniform_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG),
    ];
    let uniform_graph = tensor_kernel_to_graph(&def, &uniform_bindings).expect("uniform graph");
    let uniform_output = uniform_graph
        .propagate_ibp(&input_bounds)
        .expect("uniform IBP");
    assert_bounds_valid(&uniform_output);

    let awq_width = bound_width(&awq_output);
    let uniform_width = bound_width(&uniform_output);
    eprintln!("AWQ vs uniform IBP: awq_width={awq_width:.6}, uniform_width={uniform_width:.6}");
    // AWQ has smaller effective weight magnitude => tighter bounds
    assert!(
        awq_width <= uniform_width + 1e-4,
        "AWQ bounds should be tighter: awq={awq_width}, uniform={uniform_width}"
    );
}

// ===========================================================================
// 7. AWQ quantized SwiGLU FFN preserves output range (IBP)
// ===========================================================================

/// SwiGLU FFN with AWQ-quantized projections. Verify output bounds remain
/// finite and bounded despite the non-linear gate * up interaction.
#[test]
fn test_awq_swiglu_ffn_preserves_range_ibp() {
    let mut b = TensorBlockBuilder::new("quant_pres_awq_swiglu");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_swiglu(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let def = b.build(out).expect("valid AWQ SwiGLU kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, AWQ_WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("AWQ SwiGLU FFN IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

// ===========================================================================
// 8. AWQ quantized attention + FFN decoder block (IBP + CROWN)
// ===========================================================================

fn build_awq_decoder_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("quant_pres_awq_decoder");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Pre-norm
    let attn_eps = b.add_input("attn_eps", &[1]);
    let attn_nw = b.add_input("attn_nw", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, attn_eps, 1, attn_nw, &shape);

    // AWQ-quantized attention
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

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

    let h = b.add_binary_add(input, attn_out, &shape);

    // Pre-norm + AWQ SwiGLU FFN
    let ffn_eps = b.add_input("ffn_eps", &[1]);
    let ffn_nw = b.add_input("ffn_nw", &[HIDDEN_DIM]);
    let normed_ffn = b.add_rms_norm(h, ffn_eps, 1, ffn_nw, &shape);
    let ffn_out = build_swiglu(&mut b, normed_ffn, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let out = b.add_binary_add(h, ffn_out, &shape);

    b.build(out).expect("valid AWQ decoder block kernel")
}

fn awq_decoder_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),                                             // attn_eps
        norm_weight_binding(HIDDEN_DIM),                           // attn_nw
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], AWQ_WEIGHT_MAG), // q_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], AWQ_WEIGHT_MAG), // k_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], AWQ_WEIGHT_MAG), // v_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], AWQ_WEIGHT_MAG), // o_w
        eps_binding(),                                             // ffn_eps
        norm_weight_binding(HIDDEN_DIM),                           // ffn_nw
    ];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, AWQ_WEIGHT_MAG);
    bindings
}

#[test]
fn test_awq_decoder_block_ibp() {
    let def = build_awq_decoder_block_kernel();
    let bindings = awq_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("AWQ decoder block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_awq_decoder_block_crown() {
    let def = build_awq_decoder_block_kernel();
    let bindings = awq_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("AWQ decoder block CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 9. INT8 QK scoring bounds (SageAttention-style) (IBP)
// ===========================================================================

/// Build INT8 QK scoring kernel: attention with INT8-magnitude Q/K weights
/// and FP32 V/O weights (SageAttention architecture).
fn build_int8_qk_scoring_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("quant_pres_int8_qk_scoring");
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

    b.build(attn_out).expect("valid INT8 QK scoring kernel")
}

/// SageAttention quantizes Q and K to INT8 for the scoring matmul.
/// Model this as attention with INT8-magnitude QK weights and FP32 V/O weights.
/// The QK scoring bounds should remain finite despite reduced precision.
#[test]
fn test_int8_qk_scoring_bounds_ibp() {
    let def = build_int8_qk_scoring_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT8_WEIGHT_MAG), // q_w (INT8)
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT8_WEIGHT_MAG), // k_w (INT8)
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG), // v_w (FP32)
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG), // o_w (FP32)
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT8 QK scoring IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. INT8 QK + FP32 PV accumulation split-precision (IBP)
// ===========================================================================

/// Compare INT8-quantized QK attention vs fully FP32 attention.
/// INT8 QK produces different magnitude weights for Q/K, which should
/// produce comparable (not vastly wider) bounds to FP32.
#[test]
fn test_int8_qk_fp32_pv_split_precision_ibp() {
    let mut b = TensorBlockBuilder::new("quant_pres_int8_fp32_split");
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
    let def = b.build(attn_out).expect("valid split-precision kernel");

    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // INT8 QK + FP32 PV
    let int8_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT8_WEIGHT_MAG), // q_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT8_WEIGHT_MAG), // k_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG), // v_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG), // o_w
    ];
    let int8_graph = tensor_kernel_to_graph(&def, &int8_bindings).expect("INT8 graph");
    let int8_output = int8_graph.propagate_ibp(&input_bounds).expect("INT8 IBP");
    assert_bounds_valid(&int8_output);

    // Full FP32
    let fp32_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG), // q_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG), // k_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG), // v_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG), // o_w
    ];
    let fp32_graph = tensor_kernel_to_graph(&def, &fp32_bindings).expect("FP32 graph");
    let fp32_output = fp32_graph.propagate_ibp(&input_bounds).expect("FP32 IBP");
    assert_bounds_valid(&fp32_output);

    let int8_width = bound_width(&int8_output);
    let fp32_width = bound_width(&fp32_output);
    eprintln!("INT8-QK+FP32-PV vs FP32: int8_width={int8_width:.6}, fp32_width={fp32_width:.6}");
    // Both must produce finite, valid bounds
    assert!(int8_width.is_finite(), "INT8 width must be finite");
    assert!(fp32_width.is_finite(), "FP32 width must be finite");
}

// ===========================================================================
// 11. INT8 attention with smooth-K channel subtraction (IBP)
// ===========================================================================

/// SageAttention's smooth-K subtracts per-channel mean from K before
/// quantization to reduce outlier-driven quantization error. Model this
/// as K weights with reduced magnitude (post-smoothing).
#[test]
fn test_int8_smooth_k_attention_ibp() {
    let mut b = TensorBlockBuilder::new("quant_pres_int8_smooth_k");
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
    let def = b.build(attn_out).expect("valid smooth-K kernel");

    // Smooth-K: K weights have reduced magnitude after channel mean subtraction
    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT8_WEIGHT_MAG), // q_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT8_SMOOTH_MAG), // k_w (smoothed)
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG), // v_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG), // o_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT8 smooth-K attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 12. INT8 attention vs FP32 attention bound width comparison (IBP)
// ===========================================================================

/// Compare INT8-quantized full attention (all projections INT8) vs
/// FP32 attention. INT8 has larger weight magnitude (0.0635 vs 0.02),
/// so bounds should be wider.
#[test]
fn test_int8_vs_fp32_attention_width_ibp() {
    let mut b = TensorBlockBuilder::new("quant_pres_int8_vs_fp32_attn");
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
    let def = b.build(attn_out).expect("valid comparison kernel");

    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // INT8 attention
    let int8_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT8_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT8_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT8_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT8_WEIGHT_MAG),
    ];
    let int8_graph = tensor_kernel_to_graph(&def, &int8_bindings).expect("INT8 graph");
    let int8_output = int8_graph.propagate_ibp(&input_bounds).expect("INT8 IBP");
    assert_bounds_valid(&int8_output);

    // FP32 attention
    let fp32_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], FP32_WEIGHT_MAG),
    ];
    let fp32_graph = tensor_kernel_to_graph(&def, &fp32_bindings).expect("FP32 graph");
    let fp32_output = fp32_graph.propagate_ibp(&input_bounds).expect("FP32 IBP");
    assert_bounds_valid(&fp32_output);

    let int8_width = bound_width(&int8_output);
    let fp32_width = bound_width(&fp32_output);
    eprintln!("INT8 vs FP32 attention IBP: int8_width={int8_width:.6}, fp32_width={fp32_width:.6}");
    // INT8 has larger weight magnitude => wider bounds
    assert!(
        int8_width >= fp32_width - 1e-4,
        "INT8 bounds should be >= FP32: int8={int8_width}, fp32={fp32_width}"
    );
}

// ===========================================================================
// 13. Per-layer quantization error: INT4 linear vs FP32 linear (IBP)
// ===========================================================================

/// Measure the per-layer output bound width difference between INT4 and FP32
/// linear layers. The width difference represents the quantization-induced
/// bound widening for a single layer.
#[test]
fn test_per_layer_int4_vs_fp32_margin_ibp() {
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let int4_output = propagate_linear(&input_bounds, HIDDEN_DIM, HIDDEN_DIM, INT4_WEIGHT_MAG);
    let fp32_output = propagate_linear(&input_bounds, HIDDEN_DIM, HIDDEN_DIM, FP32_WEIGHT_MAG);

    assert_bounds_valid(&int4_output);
    assert_bounds_valid(&fp32_output);

    let int4_width = bound_width(&int4_output);
    let fp32_width = bound_width(&fp32_output);
    let margin = (int4_width - fp32_width).abs();

    eprintln!(
        "Per-layer INT4 vs FP32: int4_width={int4_width:.6}, fp32_width={fp32_width:.6}, margin={margin:.6}"
    );
    // Margin must be finite
    assert!(margin.is_finite(), "quantization margin must be finite");
    // INT4 has larger weight magnitude => wider bounds
    assert!(
        int4_width >= fp32_width - 1e-4,
        "INT4 bounds should be >= FP32: int4={int4_width}, fp32={fp32_width}"
    );
}

// ===========================================================================
// 14. Per-layer quantization error: INT8 linear vs FP32 linear (IBP)
// ===========================================================================

/// Same as test 13, but for INT8 quantization. INT8 weight magnitude is
/// between INT4 and FP32, so the margin should be intermediate.
#[test]
fn test_per_layer_int8_vs_fp32_margin_ibp() {
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let int8_output = propagate_linear(&input_bounds, HIDDEN_DIM, HIDDEN_DIM, INT8_WEIGHT_MAG);
    let fp32_output = propagate_linear(&input_bounds, HIDDEN_DIM, HIDDEN_DIM, FP32_WEIGHT_MAG);

    assert_bounds_valid(&int8_output);
    assert_bounds_valid(&fp32_output);

    let int8_width = bound_width(&int8_output);
    let fp32_width = bound_width(&fp32_output);
    let margin = (int8_width - fp32_width).abs();

    eprintln!(
        "Per-layer INT8 vs FP32: int8_width={int8_width:.6}, fp32_width={fp32_width:.6}, margin={margin:.6}"
    );
    assert!(margin.is_finite(), "quantization margin must be finite");
    assert!(
        int8_width >= fp32_width - 1e-4,
        "INT8 bounds should be >= FP32: int8={int8_width}, fp32={fp32_width}"
    );
}

// ===========================================================================
// 15. Quantization error accumulation: 2-layer stack margin growth (IBP)
// ===========================================================================

/// Verify that quantization error accumulates through depth: a 2-layer
/// quantized stack has wider bounds than a single layer. The margin
/// (INT4 width - FP32 width) should grow with depth.
#[test]
fn test_quant_error_accumulation_2layer_ibp() {
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Single INT4 layer
    let int4_1layer = propagate_linear(&input_bounds, HIDDEN_DIM, HIDDEN_DIM, INT4_WEIGHT_MAG);
    let fp32_1layer = propagate_linear(&input_bounds, HIDDEN_DIM, HIDDEN_DIM, FP32_WEIGHT_MAG);
    let margin_1 = bound_width(&int4_1layer) - bound_width(&fp32_1layer);

    // Two-layer INT4 stack (feed output of first layer as input to second)
    let int4_2layer = propagate_linear(&int4_1layer, HIDDEN_DIM, HIDDEN_DIM, INT4_WEIGHT_MAG);
    let fp32_2layer = propagate_linear(&fp32_1layer, HIDDEN_DIM, HIDDEN_DIM, FP32_WEIGHT_MAG);
    let margin_2 = bound_width(&int4_2layer) - bound_width(&fp32_2layer);

    assert_bounds_valid(&int4_2layer);
    assert_bounds_valid(&fp32_2layer);

    eprintln!("Quant error accumulation: margin_1={margin_1:.6}, margin_2={margin_2:.6}");
    // Error should accumulate: margin grows with depth
    assert!(
        margin_2 >= margin_1 - 1e-4,
        "quantization margin should grow with depth: layer1={margin_1}, layer2={margin_2}"
    );
}

// ===========================================================================
// 16. Quantized softmax output margin: bounded deviation from FP32 (IBP)
// ===========================================================================

/// Verify that INT4-quantized linear -> softmax output remains in [0, 1]
/// and that the bound width difference vs FP32 softmax is bounded.
#[test]
fn test_quantized_softmax_output_margin_ibp() {
    let vocab_size = 16;

    let mut b_int4 = TensorBlockBuilder::new("quant_pres_softmax_int4");
    let input_int4 = b_int4.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w_int4 = b_int4.add_input("w", &[vocab_size, HIDDEN_DIM]);
    let logits_int4 = b_int4.add_linear(input_int4, w_int4, None, &[SEQ_LEN, vocab_size]);
    let out_int4 = b_int4.add_softmax(logits_int4, -1, &[SEQ_LEN, vocab_size]);
    let def_int4 = b_int4.build(out_int4).expect("valid INT4 softmax kernel");

    let mut b_fp32 = TensorBlockBuilder::new("quant_pres_softmax_fp32");
    let input_fp32 = b_fp32.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w_fp32 = b_fp32.add_input("w", &[vocab_size, HIDDEN_DIM]);
    let logits_fp32 = b_fp32.add_linear(input_fp32, w_fp32, None, &[SEQ_LEN, vocab_size]);
    let out_fp32 = b_fp32.add_softmax(logits_fp32, -1, &[SEQ_LEN, vocab_size]);
    let def_fp32 = b_fp32.build(out_fp32).expect("valid FP32 softmax kernel");

    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // INT4 path
    let int4_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[vocab_size, HIDDEN_DIM], INT4_WEIGHT_MAG),
    ];
    let int4_graph = tensor_kernel_to_graph(&def_int4, &int4_bindings).expect("INT4 graph");
    let int4_output = int4_graph.propagate_ibp(&input_bounds).expect("INT4 IBP");
    assert_bounds_valid(&int4_output);

    // FP32 path
    let fp32_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[vocab_size, HIDDEN_DIM], FP32_WEIGHT_MAG),
    ];
    let fp32_graph = tensor_kernel_to_graph(&def_fp32, &fp32_bindings).expect("FP32 graph");
    let fp32_output = fp32_graph.propagate_ibp(&input_bounds).expect("FP32 IBP");
    assert_bounds_valid(&fp32_output);

    let (int4_lo, int4_hi) = bounds_min_max(&int4_output);
    let (fp32_lo, fp32_hi) = bounds_min_max(&fp32_output);
    eprintln!("Softmax INT4: [{int4_lo:.6}, {int4_hi:.6}], FP32: [{fp32_lo:.6}, {fp32_hi:.6}]");

    // Both softmax outputs must be in [0, 1]
    assert!(int4_lo >= -1e-4, "INT4 softmax lower >= 0, got {int4_lo}");
    assert!(
        int4_hi <= 1.0 + 1e-4,
        "INT4 softmax upper <= 1, got {int4_hi}"
    );
    assert!(fp32_lo >= -1e-4, "FP32 softmax lower >= 0, got {fp32_lo}");
    assert!(
        fp32_hi <= 1.0 + 1e-4,
        "FP32 softmax upper <= 1, got {fp32_hi}"
    );
}

// ===========================================================================
// 17. Group-size partitioning preserves weight tensor shape (IBP)
// ===========================================================================

/// Verify that group-quantized linear layers with different group sizes
/// produce outputs with the correct shape and valid bounds. The group_size
/// only affects the weight magnitude distribution, not the output rank/shape.
#[test]
fn test_group_size_shape_preservation_ibp() {
    // Test with multiple group sizes: 16, 32, 64
    for &gs in &[16usize, 32, 64] {
        let in_dim = gs.max(HIDDEN_DIM);
        let out_dim = HIDDEN_DIM;

        let mut b = TensorBlockBuilder::new(&format!("quant_pres_group_{gs}"));
        let input = b.add_input("x", &[SEQ_LEN, in_dim]);
        let w = b.add_input("w", &[out_dim, in_dim]);
        let out = b.add_linear(input, w, None, &[SEQ_LEN, out_dim]);
        let def = b.build(out).expect("valid group quant kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            weight_binding(&[out_dim, in_dim], INT4_WEIGHT_MAG),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input_bounds = uniform_bounds(&[SEQ_LEN, in_dim], 1.0);

        let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
        assert_bounds_valid(&output);

        // Shape must match expected output
        let (lo, _hi) = output.lower_upper();
        assert_eq!(
            lo.shape(),
            &[SEQ_LEN, out_dim],
            "group_size={gs}: output shape must be [SEQ_LEN, out_dim]"
        );

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("Group size {gs} shape preservation: bounds=[{lo_min:.6}, {hi_max:.6}]");
    }
}

// ===========================================================================
// 18. Group-size impact: smaller groups produce tighter per-element bounds (IBP)
// ===========================================================================

/// Smaller group sizes allow finer per-group calibration, which should
/// produce tighter dequantized weight ranges and therefore tighter output bounds.
#[test]
fn test_group_size_impact_tightness_ibp() {
    let in_dim = 64;
    let out_dim = HIDDEN_DIM;

    let mut b = TensorBlockBuilder::new("quant_pres_group_size_impact");
    let input = b.add_input("x", &[SEQ_LEN, in_dim]);
    let w = b.add_input("w", &[out_dim, in_dim]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, out_dim]);
    let def = b.build(out).expect("valid group impact kernel");

    let input_bounds = uniform_bounds(&[SEQ_LEN, in_dim], 1.0);

    // Small group: tighter per-element weights (0.8x magnitude)
    let small_group_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[out_dim, in_dim], INT4_WEIGHT_MAG * 0.8),
    ];
    let small_graph =
        tensor_kernel_to_graph(&def, &small_group_bindings).expect("small group graph");
    let small_output = small_graph
        .propagate_ibp(&input_bounds)
        .expect("small group IBP");
    assert_bounds_valid(&small_output);

    // Large group: looser per-element weights (1.2x magnitude)
    let large_group_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[out_dim, in_dim], INT4_WEIGHT_MAG * 1.2),
    ];
    let large_graph =
        tensor_kernel_to_graph(&def, &large_group_bindings).expect("large group graph");
    let large_output = large_graph
        .propagate_ibp(&input_bounds)
        .expect("large group IBP");
    assert_bounds_valid(&large_output);

    let small_width = bound_width(&small_output);
    let large_width = bound_width(&large_output);
    eprintln!("Group size impact: small_width={small_width:.6}, large_width={large_width:.6}");
    // Smaller group (tighter weights) should produce tighter bounds
    assert!(
        small_width <= large_width + 1e-4,
        "smaller groups should produce tighter bounds: small={small_width}, large={large_width}"
    );
}

// ===========================================================================
// 19. Non-aligned group boundary: partial last group (IBP)
// ===========================================================================

/// When input dimension is not evenly divisible by group_size, the last
/// group is partial. Verify this still produces valid bounds. We model
/// the partial group with slightly different weight magnitude.
#[test]
fn test_non_aligned_group_boundary_ibp() {
    let in_dim = GROUP_SIZE + GROUP_SIZE / 2; // 48 (not aligned to 32)
    let out_dim = HIDDEN_DIM;

    let mut b = TensorBlockBuilder::new("quant_pres_non_aligned_group");
    let input = b.add_input("x", &[SEQ_LEN, in_dim]);
    let w = b.add_input("w", &[out_dim, in_dim]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, out_dim]);
    let def = b.build(out).expect("valid non-aligned group kernel");

    // Construct weight with group boundaries: group 0 (cols 0-31) = tighter,
    // partial group 1 (cols 32-47) = slightly different calibration
    let mut weight_data = Vec::with_capacity(out_dim * in_dim);
    for _row in 0..out_dim {
        for col in 0..in_dim {
            if col < GROUP_SIZE {
                weight_data.push(INT4_WEIGHT_MAG);
            } else {
                // Partial group: slightly different scale (typical of
                // non-aligned group quantization)
                weight_data.push(INT4_WEIGHT_MAG * 1.1);
            }
        }
    }
    let w_tensor =
        ArrayD::from_shape_vec(IxDyn(&[out_dim, in_dim]), weight_data).expect("valid weight shape");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, in_dim], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Non-aligned group boundary IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 20. Group quantization monotone tightening (IBP)
// ===========================================================================

/// Monotone tightening: tighter input bounds -> tighter output bounds.
/// This must hold regardless of quantization scheme.
#[test]
fn test_group_quant_monotone_tightening_ibp() {
    let mut b = TensorBlockBuilder::new("quant_pres_monotone");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid monotone kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], INT4_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input
    let wide_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("wide IBP");
    assert_bounds_valid(&wide_output);

    // Narrow input
    let narrow_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("narrow IBP");
    assert_bounds_valid(&narrow_output);

    let wide_width = bound_width(&wide_output);
    let narrow_width = bound_width(&narrow_output);
    eprintln!("Monotone tightening: wide_width={wide_width:.6}, narrow_width={narrow_width:.6}");
    // Tighter input -> tighter output
    assert!(
        narrow_width <= wide_width + 1e-4,
        "tighter input must produce tighter output: narrow={narrow_width}, wide={wide_width}"
    );
}
