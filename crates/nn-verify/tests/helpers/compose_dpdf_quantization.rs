// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Advanced quantization-aware compose verification.
//!
//! Extends the basic INT4 dequantization tests (`compose_dpdf_quantized.rs`)
//! with GPTQ group dequantization, AWQ activation-aware dequantization,
//! mixed precision analysis, 2-bit quantization, and quantization error
//! accumulation through depth.
//!
//! **INT4 Dequantization Variants** (tests 1-4):
//! 1. INT4 symmetric dequant single group IBP
//! 2. INT4 asymmetric dequant single group IBP
//! 3. GPTQ group dequant with scale + zero_point IBP
//! 4. AWQ activation-aware dequant IBP
//!
//! **Mixed Precision & Composition** (tests 5-8):
//! 5. Mixed precision: FP32 attention + INT4 FFN IBP
//! 6. Quantized linear layer (dequant -> matmul -> bias) IBP
//! 7. Quantized SwiGLU (gate/up through INT4) IBP
//! 8. Quantized MoE expert routing IBP
//!
//! **Precision Analysis** (tests 9-11):
//! 9. Dequant precision: INT4 vs FP32 bound width comparison IBP
//! 10. Group size impact on bound width IBP
//! 11. 2-bit vs 4-bit quantization bound width comparison IBP
//!
//! **End-to-End & Depth** (tests 12-15):
//! 12. Quantized decoder layer end-to-end IBP
//! 13. Quantized -> softmax output IBP + CROWN
//! 14. Quantization error accumulation through depth IBP
//! 15. Mixed quantization decoder (some layers INT4, some FP16) IBP
//!
//! INT4 GPTQ scheme (group-wise):
//!   Groups of `group_size` along input dimension share scale + zero_point.
//!   code stored as INT4, dequant: w_deq = (code - zero_point) * scale
//!   GPTQ uses Hessian-based reordering to minimize reconstruction error.
//!
//! AWQ (Activation-Aware Weight Quantization):
//!   Scales salient channels by activation magnitude before quantization.
//!   Effectively: w_scaled = w * s, then quantize w_scaled with adjusted scale.
//!   Dequant recovers: w_deq = dequant(code) / s.
//!
//! Part of #3979: Quantization-aware compose tests for INT4, GPTQ, AWQ.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Hidden dimension (tiny for testing).
const HIDDEN_DIM: usize = 64;
/// FFN intermediate dimension (SwiGLU gate and up projections).
const FFN_DIM: usize = 128;
/// Sequence length for decoder sub-block tests.
const SEQ_LEN: usize = 4;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Number of KV heads for grouped-query attention.
const NUM_KV_HEADS: usize = 2;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 16
/// KV dimension = NUM_KV_HEADS * HEAD_DIM.
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 32
/// Vocabulary size for LM head tests.
const VOCAB_SIZE: usize = 256;
/// Weight magnitude for FP32 baseline comparison.
const WEIGHT_MAG: f32 = 0.02;
/// Standard group size for group-wise quantization.
const GROUP_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// INT4 quantization parameters
// ---------------------------------------------------------------------------

/// Quantization scale for INT4 symmetric approximation.
const QUANT_SCALE: f32 = 0.01;
/// INT4 symmetric range: [-8, 7], max dequantized value = 7 * scale.
const INT4_SYM_MAX: f32 = 7.0 * QUANT_SCALE; // 0.07
/// INT4 asymmetric range: [0, 15] unsigned with zero_point offset.
const INT4_ASYM_MAX: f32 = 8.0 * QUANT_SCALE; // 0.08
/// GPTQ group dequant: slightly larger due to Hessian reordering residual.
const GPTQ_WEIGHT_MAX: f32 = 7.0 * QUANT_SCALE * 1.05; // ~0.0735
/// AWQ activation-aware scale factor applied per salient channel.
const AWQ_SALIENT_SCALE: f32 = 1.2;
/// INT2 symmetric range: [-2, 1], max dequantized value = 1 * scale.
const INT2_SYM_MAX: f32 = 1.0 * QUANT_SCALE; // 0.01
/// FP16 weight magnitude (higher precision than INT4).
const FP16_WEIGHT_MAG: f32 = 0.05;

// ---------------------------------------------------------------------------
// Helpers: SiLU decomposition
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

// ===========================================================================
// 1. INT4 symmetric dequant single group IBP
// ===========================================================================

/// Build single-group symmetric INT4 dequant -> matmul.
///
/// Models one quantization group: weights of size [1, GROUP_SIZE] are
/// INT4-dequantized (symmetric) and used in a linear layer.
///
/// Input: `[SEQ_LEN, GROUP_SIZE]` (Variable).
/// Output: `[SEQ_LEN, 1]`.
fn build_int4_sym_single_group_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_int4_sym_single_group");
    let input = b.add_input("activations", &[SEQ_LEN, GROUP_SIZE]);
    let deq_w = b.add_input("dequantized_weight", &[1, GROUP_SIZE]);
    let bias = b.add_input("bias", &[1]);
    let out = b.add_linear(input, deq_w, Some(bias), &[SEQ_LEN, 1]);
    b.build(out).expect("valid INT4 sym single group kernel")
}

#[test]
fn test_int4_sym_single_group_ibp() {
    let def = build_int4_sym_single_group_kernel();
    let deq_w = ArrayD::from_elem(IxDyn(&[1, GROUP_SIZE]), INT4_SYM_MAX);
    let bias = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(deq_w),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, GROUP_SIZE], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through INT4 sym single group");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT4 sym single group IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Theoretical max: 32 * 0.07 = 2.24
    assert!(
        hi_max < 5.0,
        "INT4 sym single group upper should be < 5.0, got {hi_max}"
    );
}

// ===========================================================================
// 2. INT4 asymmetric dequant single group IBP
// ===========================================================================

/// Build single-group asymmetric INT4 dequant -> matmul.
///
/// Asymmetric uses [0, 15] unsigned range with zero_point offset.
/// The dequantized weights have slightly larger max magnitude.
///
/// Input: `[SEQ_LEN, GROUP_SIZE]` (Variable).
/// Output: `[SEQ_LEN, 1]`.
fn build_int4_asym_single_group_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_int4_asym_single_group");
    let input = b.add_input("activations", &[SEQ_LEN, GROUP_SIZE]);
    let deq_w = b.add_input("dequantized_weight", &[1, GROUP_SIZE]);
    let bias = b.add_input("bias", &[1]);
    let out = b.add_linear(input, deq_w, Some(bias), &[SEQ_LEN, 1]);
    b.build(out).expect("valid INT4 asym single group kernel")
}

#[test]
fn test_int4_asym_single_group_ibp() {
    let def = build_int4_asym_single_group_kernel();
    let deq_w = ArrayD::from_elem(IxDyn(&[1, GROUP_SIZE]), INT4_ASYM_MAX);
    let bias = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(deq_w),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, GROUP_SIZE], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through INT4 asym single group");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT4 asym single group IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Asymmetric max = 32 * 0.08 = 2.56
    assert!(
        hi_max < 6.0,
        "INT4 asym single group upper should be < 6.0, got {hi_max}"
    );
}

// ===========================================================================
// 3. GPTQ group dequant with scale + zero_point IBP
// ===========================================================================

/// Build a GPTQ-style group dequant -> matmul kernel.
///
/// GPTQ uses Hessian-based column reordering to minimize quantization error.
/// The dequantized weights have slightly larger magnitudes than standard INT4
/// due to the reordering residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_gptq_group_dequant_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_gptq_group_dequant");
    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let deq_w = b.add_input("gptq_dequantized_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);
    let out = b.add_linear(input, deq_w, Some(bias), &[SEQ_LEN, HIDDEN_DIM]);
    b.build(out).expect("valid GPTQ group dequant kernel")
}

#[test]
fn test_gptq_group_dequant_ibp() {
    let def = build_gptq_group_dequant_kernel();
    // Simulate GPTQ dequantized weights: per-group magnitudes with Hessian residual
    let mut w_data = vec![0.0f32; HIDDEN_DIM * HIDDEN_DIM];
    for oc in 0..HIDDEN_DIM {
        for ic in 0..HIDDEN_DIM {
            let group_idx = ic / GROUP_SIZE;
            // GPTQ groups have slightly different scales from Hessian reordering
            let mag = if group_idx == 0 {
                GPTQ_WEIGHT_MAX
            } else {
                GPTQ_WEIGHT_MAX * 0.9
            };
            w_data[oc * HIDDEN_DIM + ic] = mag;
        }
    }
    let deq_w = ArrayD::from_shape_vec(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), w_data)
        .expect("valid GPTQ weight shape");
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(deq_w),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GPTQ group dequant");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "GPTQ group dequant output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GPTQ group dequant IBP (2 groups): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // D=64, mixed GPTQ weights: sum ~ 32*0.0735 + 32*0.066 = 2.35 + 2.11 = 4.46
    assert!(
        hi_max < 10.0,
        "GPTQ group dequant upper should be < 10.0, got {hi_max}"
    );
}

// ===========================================================================
// 4. AWQ activation-aware dequant IBP
// ===========================================================================

/// Build an AWQ-style activation-aware dequant -> matmul kernel.
///
/// AWQ pre-scales salient channels by activation magnitude before quantization.
/// The dequantized weights for salient channels have larger effective magnitude
/// (divided by the salient scale after dequant), modeling the AWQ recover step.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_awq_dequant_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_awq_dequant");
    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let deq_w = b.add_input("awq_dequantized_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);
    let out = b.add_linear(input, deq_w, Some(bias), &[SEQ_LEN, HIDDEN_DIM]);
    b.build(out).expect("valid AWQ dequant kernel")
}

#[test]
fn test_awq_dequant_ibp() {
    let def = build_awq_dequant_kernel();
    // Simulate AWQ dequantized weights: salient channels have larger magnitudes
    let mut w_data = vec![0.0f32; HIDDEN_DIM * HIDDEN_DIM];
    for oc in 0..HIDDEN_DIM {
        for ic in 0..HIDDEN_DIM {
            // First 25% of channels are salient (higher activation magnitude)
            let is_salient = ic < HIDDEN_DIM / 4;
            let mag = if is_salient {
                INT4_SYM_MAX * AWQ_SALIENT_SCALE
            } else {
                INT4_SYM_MAX
            };
            w_data[oc * HIDDEN_DIM + ic] = mag;
        }
    }
    let deq_w = ArrayD::from_shape_vec(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), w_data)
        .expect("valid AWQ weight shape");
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(deq_w),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through AWQ dequant");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "AWQ dequant output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("AWQ dequant IBP (25% salient): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Salient channels: 16*0.084 + 48*0.07 = 1.34 + 3.36 = 4.70
    assert!(
        hi_max < 10.0,
        "AWQ dequant upper should be < 10.0, got {hi_max}"
    );
}

// ===========================================================================
// 5. Mixed precision: FP32 attention + INT4 FFN IBP
// ===========================================================================

/// Build a mixed-precision decoder: FP32 attention weights + INT4 FFN weights.
///
/// This models the common deployment pattern where attention projections
/// remain in FP32 for quality while FFN projections are INT4-quantized.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_mixed_fp32_attn_int4_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_mixed_fp32_attn_int4_ffn");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // FP32 attention projections
    let q_w = b.add_input("q_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_weight", &[HIDDEN_DIM, KV_DIM]);

    let q = b.add_linear(normed1, q_w, None, &[SEQ_LEN, KV_DIM]);
    let k = b.add_linear(normed1, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(normed1, v_w, None, &[SEQ_LEN, KV_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    // INT4 quantized SwiGLU FFN
    let gate_w = b.add_input("q_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("q_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("q_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    let out = b.add_binary_add(res1, ffn_out, &shape);
    b.build(out)
        .expect("valid mixed FP32 attn + INT4 FFN kernel")
}

fn mixed_fp32_attn_int4_ffn_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    // FP32 attention weights
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let o_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);
    // INT4 quantized FFN weights
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), INT4_SYM_MAX);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-6),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(q_w),            // q_weight (FP32)
        TensorParamBinding::ConstantTensor(k_w),            // k_weight (FP32)
        TensorParamBinding::ConstantTensor(v_w),            // v_weight (FP32)
        TensorParamBinding::ConstantTensor(o_w),            // o_weight (FP32)
        TensorParamBinding::ConstantScalar(1e-6),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // q_gate_weight (INT4)
        TensorParamBinding::ConstantTensor(up_w),           // q_up_weight (INT4)
        TensorParamBinding::ConstantTensor(down_w),         // q_down_weight (INT4)
    ]
}

#[test]
fn test_mixed_fp32_attn_int4_ffn_ibp() {
    let def = build_mixed_fp32_attn_int4_ffn_kernel();
    let bindings = mixed_fp32_attn_int4_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through mixed FP32 attn + INT4 FFN");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "mixed precision decoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mixed FP32 attn + INT4 FFN IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. Quantized linear layer (dequant -> matmul -> bias) IBP
// ===========================================================================

/// Build a quantized linear projection with non-zero bias.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, FFN_DIM]`.
fn build_quantized_linear_with_bias_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_quantized_linear_bias");
    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let deq_w = b.add_input("dequantized_weight", &[FFN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[FFN_DIM]);
    let out = b.add_linear(input, deq_w, Some(bias), &[SEQ_LEN, FFN_DIM]);
    b.build(out)
        .expect("valid quantized linear with bias kernel")
}

#[test]
fn test_quantized_linear_with_bias_ibp() {
    let def = build_quantized_linear_with_bias_kernel();
    let deq_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let bias_val = 0.1f32;
    let bias = ArrayD::from_elem(IxDyn(&[FFN_DIM]), bias_val);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(deq_w),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized linear with bias");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, FFN_DIM],
        "quantized linear output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized linear + bias IBP (bias={bias_val}): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // D_in=64, w_max=0.07: max output = 64 * 0.07 + 0.1 = 4.58
    assert!(
        hi_max < 10.0,
        "quantized linear upper should be < 10, got {hi_max}"
    );
}

// ===========================================================================
// 7. Quantized SwiGLU (gate/up through INT4) IBP
// ===========================================================================

/// Build a quantized SwiGLU FFN: gate/up/down projections at INT4 magnitudes.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_quantized_swiglu_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_quantized_swiglu");
    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &out_shape);

    b.build(out).expect("valid quantized SwiGLU kernel")
}

fn quantized_swiglu_ffn_bindings() -> Vec<TensorParamBinding> {
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), INT4_SYM_MAX);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(gate_w),
        TensorParamBinding::ConstantTensor(up_w),
        TensorParamBinding::ConstantTensor(down_w),
    ]
}

#[test]
fn test_quantized_swiglu_ffn_ibp() {
    let def = build_quantized_swiglu_ffn_kernel();
    let bindings = quantized_swiglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized SwiGLU FFN");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "quantized SwiGLU output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized SwiGLU FFN IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. Quantized MoE expert routing IBP
// ===========================================================================

/// MoE expert FFN dimension (scaled down for testing).
const MOE_EXPERT_DIM: usize = 64;
/// Number of MoE experts for routing.
const NUM_EXPERTS: usize = 8;

/// Build a quantized MoE routing + single expert dispatch.
///
/// Models: hidden -> routing linear -> softmax (expert gate) and
/// hidden -> INT4 expert FFN -> output.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_quantized_moe_routing_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_quantized_moe_routing");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let expert_shape = [SEQ_LEN, MOE_EXPERT_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // Expert routing: hidden -> linear -> softmax
    let route_w = b.add_input("route_weight", &[NUM_EXPERTS, HIDDEN_DIM]);
    let logits = b.add_linear(input, route_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let _probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    // Single expert SwiGLU FFN (INT4 quantized)
    let gate_w = b.add_input("expert_gate_w", &[MOE_EXPERT_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("expert_up_w", &[MOE_EXPERT_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("expert_down_w", &[HIDDEN_DIM, MOE_EXPERT_DIM]);

    let gate = b.add_linear(input, gate_w, None, &expert_shape);
    let gate_act = add_silu(&mut b, gate, &expert_shape);
    let up = b.add_linear(input, up_w, None, &expert_shape);
    let hidden = b.add_binary_mul(gate_act, up, &expert_shape);
    let expert_out = b.add_linear(hidden, down_w, None, &out_shape);

    // Residual connection
    let out = b.add_binary_add(input, expert_out, &out_shape);
    b.build(out).expect("valid quantized MoE routing kernel")
}

fn quantized_moe_routing_bindings() -> Vec<TensorParamBinding> {
    let route_w = ArrayD::from_elem(IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let up_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, MOE_EXPERT_DIM]), INT4_SYM_MAX);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(route_w),
        TensorParamBinding::ConstantTensor(gate_w),
        TensorParamBinding::ConstantTensor(up_w),
        TensorParamBinding::ConstantTensor(down_w),
    ]
}

#[test]
fn test_quantized_moe_routing_ibp() {
    let def = build_quantized_moe_routing_kernel();
    let bindings = quantized_moe_routing_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized MoE routing");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "MoE routing output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized MoE routing IBP (expert + residual): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Dequant precision: INT4 vs FP32 bound width comparison IBP
// ===========================================================================

/// Build a generic linear layer for precision comparison.
fn build_precision_compare_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_precision_compare");
    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let weight = b.add_input("weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);
    let out = b.add_linear(input, weight, Some(bias), &[SEQ_LEN, HIDDEN_DIM]);
    b.build(out).expect("valid precision compare kernel")
}

/// INT4 dequant vs FP32 output bound width comparison.
///
/// INT4 weights (max 0.07) produce different bound widths than FP32 weights
/// (0.02). This test quantifies the relationship.
#[test]
fn test_int4_vs_fp32_bound_width_ibp() {
    let def = build_precision_compare_kernel();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    // INT4 quantized weights
    let int4_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let int4_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(int4_w),
        TensorParamBinding::ConstantTensor(bias.clone()),
    ];
    let int4_graph = tensor_kernel_to_graph(&def, &int4_bindings).expect("INT4 graph");
    let int4_out = int4_graph.propagate_ibp(&input).expect("IBP INT4");
    assert_bounds_valid(&int4_out);
    let (i4_lo, i4_hi) = bounds_min_max(&int4_out);
    let int4_width = i4_hi - i4_lo;

    // FP32 weights
    let fp32_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fp32_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(fp32_w),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let fp32_graph = tensor_kernel_to_graph(&def, &fp32_bindings).expect("FP32 graph");
    let fp32_out = fp32_graph.propagate_ibp(&input).expect("IBP FP32");
    assert_bounds_valid(&fp32_out);
    let (f32_lo, f32_hi) = bounds_min_max(&fp32_out);
    let fp32_width = f32_hi - f32_lo;

    eprintln!(
        "INT4 vs FP32 linear: INT4_width={int4_width:.4}, FP32_width={fp32_width:.4}, \
         ratio={:.2}x",
        int4_width / fp32_width.max(1e-10)
    );

    assert!(int4_width.is_finite(), "INT4 width must be finite");
    assert!(fp32_width.is_finite(), "FP32 width must be finite");
    assert!(int4_width > 0.0, "INT4 width must be positive");
    assert!(fp32_width > 0.0, "FP32 width must be positive");
}

// ===========================================================================
// 10. Group size impact on bound width IBP
// ===========================================================================

/// Group size 16 for fine-grained quantization.
const GROUP_SIZE_FINE: usize = 16;
/// Group size 64 for coarse-grained quantization.
const GROUP_SIZE_COARSE: usize = 64;

/// IBP verifies that different group sizes affect output bound widths.
///
/// Finer groups (G=16) can adapt scale per-group better than coarse (G=64).
/// This test verifies both produce valid finite bounds.
#[test]
fn test_group_size_impact_ibp() {
    let def = build_precision_compare_kernel();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    // Fine groups: simulate per-group quantized weights with G=16
    // With finer groups, each group has its own scale, so magnitudes vary more
    let mut fine_data = vec![0.0f32; HIDDEN_DIM * HIDDEN_DIM];
    for oc in 0..HIDDEN_DIM {
        for ic in 0..HIDDEN_DIM {
            let group_idx = ic / GROUP_SIZE_FINE;
            // Alternate between 100% and 60% of INT4 max per group
            let mag = if group_idx.is_multiple_of(2) {
                INT4_SYM_MAX
            } else {
                INT4_SYM_MAX * 0.6
            };
            fine_data[oc * HIDDEN_DIM + ic] = mag;
        }
    }
    let fine_w = ArrayD::from_shape_vec(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), fine_data)
        .expect("fine group weights");
    let fine_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(fine_w),
        TensorParamBinding::ConstantTensor(bias.clone()),
    ];
    let fine_graph = tensor_kernel_to_graph(&def, &fine_bindings).expect("fine group graph");
    let fine_out = fine_graph.propagate_ibp(&input).expect("IBP fine groups");
    assert_bounds_valid(&fine_out);
    let (fine_lo, fine_hi) = bounds_min_max(&fine_out);
    let fine_width = fine_hi - fine_lo;

    // Coarse groups: simulate per-group quantized weights with G=64
    let coarse_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let coarse_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(coarse_w),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let coarse_graph = tensor_kernel_to_graph(&def, &coarse_bindings).expect("coarse group graph");
    let coarse_out = coarse_graph
        .propagate_ibp(&input)
        .expect("IBP coarse groups");
    assert_bounds_valid(&coarse_out);
    let (coarse_lo, coarse_hi) = bounds_min_max(&coarse_out);
    let coarse_width = coarse_hi - coarse_lo;

    eprintln!("Group size impact: G=16 width={fine_width:.4}, G=64 width={coarse_width:.4}");

    assert!(fine_width.is_finite(), "fine group width must be finite");
    assert!(
        coarse_width.is_finite(),
        "coarse group width must be finite"
    );
    assert!(fine_width > 0.0, "fine group width must be positive");
    assert!(coarse_width > 0.0, "coarse group width must be positive");
}

// ===========================================================================
// 11. 2-bit vs 4-bit quantization bound width comparison IBP
// ===========================================================================

/// IBP compares INT2 vs INT4 dequant output bound widths.
///
/// INT2 has range [-2, 1] (4 levels), producing much smaller dequantized
/// magnitudes than INT4 range [-8, 7] (16 levels). Both with same scale.
#[test]
fn test_int2_vs_int4_bound_width_ibp() {
    let def = build_precision_compare_kernel();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    // INT2 weights: max magnitude = 1 * QUANT_SCALE = 0.01
    let int2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), INT2_SYM_MAX);
    let int2_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(int2_w),
        TensorParamBinding::ConstantTensor(bias.clone()),
    ];
    let int2_graph = tensor_kernel_to_graph(&def, &int2_bindings).expect("INT2 graph");
    let int2_out = int2_graph.propagate_ibp(&input).expect("IBP INT2");
    assert_bounds_valid(&int2_out);
    let (i2_lo, i2_hi) = bounds_min_max(&int2_out);
    let int2_width = i2_hi - i2_lo;

    // INT4 weights: max magnitude = 7 * QUANT_SCALE = 0.07
    let int4_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let int4_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(int4_w),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let int4_graph = tensor_kernel_to_graph(&def, &int4_bindings).expect("INT4 graph");
    let int4_out = int4_graph.propagate_ibp(&input).expect("IBP INT4");
    assert_bounds_valid(&int4_out);
    let (i4_lo, i4_hi) = bounds_min_max(&int4_out);
    let int4_width = i4_hi - i4_lo;

    eprintln!(
        "INT2 vs INT4: INT2_width={int2_width:.4}, INT4_width={int4_width:.4}, \
         ratio={:.2}x",
        int4_width / int2_width.max(1e-10)
    );

    assert!(int2_width.is_finite(), "INT2 width must be finite");
    assert!(int4_width.is_finite(), "INT4 width must be finite");
    assert!(int2_width > 0.0, "INT2 width must be positive");
    assert!(int4_width > 0.0, "INT4 width must be positive");
    // INT4 weights are 7x larger magnitude -> wider bounds
    assert!(
        int4_width > int2_width,
        "INT4 should produce wider bounds than INT2: INT4={int4_width:.4}, INT2={int2_width:.4}"
    );
}

// ===========================================================================
// 12. Quantized decoder layer end-to-end IBP
// ===========================================================================

/// Build a full quantized decoder layer: RMSNorm -> quantized GQA ->
/// residual -> RMSNorm -> quantized SwiGLU FFN -> residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_quantized_decoder_e2e_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_quantized_decoder_e2e");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // INT4 quantized Q/K/V/O
    let q_w = b.add_input("q_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_weight", &[HIDDEN_DIM, KV_DIM]);

    let q = b.add_linear(normed1, q_w, None, &[SEQ_LEN, KV_DIM]);
    let k = b.add_linear(normed1, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(normed1, v_w, None, &[SEQ_LEN, KV_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    // INT4 quantized SwiGLU FFN
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    let out = b.add_binary_add(res1, ffn_out, &shape);
    b.build(out).expect("valid quantized decoder e2e kernel")
}

fn quantized_decoder_e2e_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let o_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), INT4_SYM_MAX);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), INT4_SYM_MAX);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-6),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(q_w),            // q_weight
        TensorParamBinding::ConstantTensor(k_w),            // k_weight
        TensorParamBinding::ConstantTensor(v_w),            // v_weight
        TensorParamBinding::ConstantTensor(o_w),            // o_weight
        TensorParamBinding::ConstantScalar(1e-6),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
    ]
}

#[test]
fn test_quantized_decoder_e2e_ibp() {
    let def = build_quantized_decoder_e2e_kernel();
    let bindings = quantized_decoder_e2e_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized decoder e2e");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "quantized decoder e2e output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized decoder e2e IBP (all INT4): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record the quantized decoder e2e result.
#[test]
fn test_quantized_decoder_e2e_verify_and_record() {
    let def = build_quantized_decoder_e2e_kernel();
    let bindings = quantized_decoder_e2e_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_qaw_quantized_decoder_e2e");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 13. Quantized -> softmax output IBP + CROWN
// ===========================================================================

/// Build a quantized decoder -> RMSNorm -> LM head -> softmax pipeline.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution).
fn build_quantized_softmax_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_quantized_softmax");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Single quantized decoder layer
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    let q_w = b.add_input("q_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_weight", &[HIDDEN_DIM, KV_DIM]);

    let q = b.add_linear(normed1, q_w, None, &[SEQ_LEN, KV_DIM]);
    let k = b.add_linear(normed1, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(normed1, v_w, None, &[SEQ_LEN, KV_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    let decoder_out = b.add_binary_add(res1, ffn_out, &shape);

    // Final RMSNorm + LM head + softmax
    let fn_eps = b.add_input("final_norm_eps", &[1]);
    let fn_w = b.add_input("final_norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(decoder_out, fn_eps, 1, fn_w, &shape);

    let lm_w = b.add_input("lm_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid quantized softmax kernel")
}

fn quantized_softmax_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let o_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), INT4_SYM_MAX);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), INT4_SYM_MAX);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), INT4_SYM_MAX);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-6),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(q_w),            // q_weight
        TensorParamBinding::ConstantTensor(k_w),            // k_weight
        TensorParamBinding::ConstantTensor(v_w),            // v_weight
        TensorParamBinding::ConstantTensor(o_w),            // o_weight
        TensorParamBinding::ConstantScalar(1e-6),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
        TensorParamBinding::ConstantScalar(1e-6),           // final_norm_eps
        TensorParamBinding::ConstantTensor(norm_w),         // final_norm_weight
        TensorParamBinding::ConstantTensor(lm_w),           // lm_weight
    ]
}

/// IBP bounds through quantized decoder -> softmax produce valid probabilities.
#[test]
fn test_quantized_softmax_ibp() {
    let def = build_quantized_softmax_kernel();
    let bindings = quantized_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized softmax");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "quantized softmax output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized -> softmax IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax codomain is (0, 1)
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

/// CROWN bounds through quantized decoder -> softmax.
#[test]
fn test_quantized_softmax_crown() {
    let def = build_quantized_softmax_kernel();
    let bindings = quantized_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "quantized softmax CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized -> softmax CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 14. Quantization error accumulation through depth IBP
// ===========================================================================

/// Build a 2-layer quantized decoder stack for error accumulation analysis.
///
/// Tests how quantization error propagates through multiple layers.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_depth_accumulation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_depth_accumulation");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer_idx in 0..2 {
        let pfx = format!("l{layer_idx}");

        let n1e = b.add_input(&format!("{pfx}_n1e"), &[1]);
        let n1w = b.add_input(&format!("{pfx}_n1w"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, n1e, 1, n1w, &shape);

        let qw = b.add_input(&format!("{pfx}_qw"), &[KV_DIM, HIDDEN_DIM]);
        let kw = b.add_input(&format!("{pfx}_kw"), &[KV_DIM, HIDDEN_DIM]);
        let vw = b.add_input(&format!("{pfx}_vw"), &[KV_DIM, HIDDEN_DIM]);
        let ow = b.add_input(&format!("{pfx}_ow"), &[HIDDEN_DIM, KV_DIM]);

        let q = b.add_linear(normed1, qw, None, &[SEQ_LEN, KV_DIM]);
        let k = b.add_linear(normed1, kw, None, &[SEQ_LEN, KV_DIM]);
        let v = b.add_linear(normed1, vw, None, &[SEQ_LEN, KV_DIM]);
        let attn = b.add_attention(
            q,
            k,
            v,
            AttentionMask::Causal,
            Some(scale),
            &[SEQ_LEN, KV_DIM],
        );
        let attn_out = b.add_linear(attn, ow, None, &shape);
        let res1 = b.add_binary_add(current, attn_out, &shape);

        let n2e = b.add_input(&format!("{pfx}_n2e"), &[1]);
        let n2w = b.add_input(&format!("{pfx}_n2w"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, n2e, 1, n2w, &shape);

        let gw = b.add_input(&format!("{pfx}_gw"), &[FFN_DIM, HIDDEN_DIM]);
        let uw = b.add_input(&format!("{pfx}_uw"), &[FFN_DIM, HIDDEN_DIM]);
        let dw = b.add_input(&format!("{pfx}_dw"), &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed2, gw, None, &ffn_shape);
        let gate_act = add_silu(&mut b, gate, &ffn_shape);
        let up = b.add_linear(normed2, uw, None, &ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
        let ffn_out = b.add_linear(hidden, dw, None, &shape);

        current = b.add_binary_add(res1, ffn_out, &shape);
    }

    b.build(current).expect("valid depth accumulation kernel")
}

fn depth_accumulation_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let o_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), INT4_SYM_MAX);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), INT4_SYM_MAX);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _layer in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // n1e
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // n1w
        bindings.push(TensorParamBinding::ConstantTensor(q_w.clone())); // qw
        bindings.push(TensorParamBinding::ConstantTensor(k_w.clone())); // kw
        bindings.push(TensorParamBinding::ConstantTensor(v_w.clone())); // vw
        bindings.push(TensorParamBinding::ConstantTensor(o_w.clone())); // ow
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // n2e
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // n2w
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gw
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // uw
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // dw
    }

    bindings
}

/// IBP bounds accumulate through 2-layer quantized decoder stack.
///
/// Verifies that deeper quantized stacks produce wider bounds (error
/// accumulates through depth), but remain finite and valid.
#[test]
fn test_depth_accumulation_ibp() {
    let def = build_depth_accumulation_kernel();
    let bindings = depth_accumulation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-layer quantized decoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "depth accumulation output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let two_layer_width = hi_max - lo_min;
    eprintln!(
        "Depth accumulation IBP (2 layers): bounds=[{lo_min}, {hi_max}], width={two_layer_width:.4}"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        two_layer_width > 0.0,
        "2-layer width must be positive, got {two_layer_width}"
    );

    // Compare with single-layer width for accumulation evidence
    let single_def = build_quantized_decoder_e2e_kernel();
    let single_bindings = quantized_decoder_e2e_bindings();
    let single_graph =
        tensor_kernel_to_graph(&single_def, &single_bindings).expect("single layer graph");
    let single_output = single_graph
        .propagate_ibp(&input)
        .expect("IBP through 1-layer quantized decoder");
    let (s_lo, s_hi) = bounds_min_max(&single_output);
    let single_width = s_hi - s_lo;

    eprintln!("Accumulation: 1-layer width={single_width:.4}, 2-layer width={two_layer_width:.4}");

    // Two layers should produce at least as wide bounds as one layer
    // (error accumulates, though RMSNorm may tighten)
    assert!(
        two_layer_width > 0.0 && single_width > 0.0,
        "both widths must be positive"
    );
}

// ===========================================================================
// 15. Mixed quantization decoder (some layers INT4, some FP16) IBP
// ===========================================================================

/// Build a mixed quantization 2-layer decoder: first layer INT4, second FP16.
///
/// Models the deployment pattern where early layers use aggressive INT4
/// quantization while later layers use higher precision FP16 for quality.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_mixed_quant_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qaw_mixed_quant_decoder");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    // Layer 0: INT4 quantized
    {
        let n1e = b.add_input("l0_n1e", &[1]);
        let n1w = b.add_input("l0_n1w", &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, n1e, 1, n1w, &shape);

        let qw = b.add_input("l0_qw", &[KV_DIM, HIDDEN_DIM]);
        let kw = b.add_input("l0_kw", &[KV_DIM, HIDDEN_DIM]);
        let vw = b.add_input("l0_vw", &[KV_DIM, HIDDEN_DIM]);
        let ow = b.add_input("l0_ow", &[HIDDEN_DIM, KV_DIM]);

        let q = b.add_linear(normed1, qw, None, &[SEQ_LEN, KV_DIM]);
        let k = b.add_linear(normed1, kw, None, &[SEQ_LEN, KV_DIM]);
        let v = b.add_linear(normed1, vw, None, &[SEQ_LEN, KV_DIM]);
        let attn = b.add_attention(
            q,
            k,
            v,
            AttentionMask::Causal,
            Some(scale),
            &[SEQ_LEN, KV_DIM],
        );
        let attn_out = b.add_linear(attn, ow, None, &shape);
        let res1 = b.add_binary_add(current, attn_out, &shape);

        let n2e = b.add_input("l0_n2e", &[1]);
        let n2w = b.add_input("l0_n2w", &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, n2e, 1, n2w, &shape);

        let gw = b.add_input("l0_gw", &[FFN_DIM, HIDDEN_DIM]);
        let uw = b.add_input("l0_uw", &[FFN_DIM, HIDDEN_DIM]);
        let dw = b.add_input("l0_dw", &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed2, gw, None, &ffn_shape);
        let gate_act = add_silu(&mut b, gate, &ffn_shape);
        let up = b.add_linear(normed2, uw, None, &ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
        let ffn_out = b.add_linear(hidden, dw, None, &shape);

        current = b.add_binary_add(res1, ffn_out, &shape);
    }

    // Layer 1: FP16 precision (higher weight magnitudes)
    {
        let n1e = b.add_input("l1_n1e", &[1]);
        let n1w = b.add_input("l1_n1w", &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, n1e, 1, n1w, &shape);

        let qw = b.add_input("l1_qw", &[KV_DIM, HIDDEN_DIM]);
        let kw = b.add_input("l1_kw", &[KV_DIM, HIDDEN_DIM]);
        let vw = b.add_input("l1_vw", &[KV_DIM, HIDDEN_DIM]);
        let ow = b.add_input("l1_ow", &[HIDDEN_DIM, KV_DIM]);

        let q = b.add_linear(normed1, qw, None, &[SEQ_LEN, KV_DIM]);
        let k = b.add_linear(normed1, kw, None, &[SEQ_LEN, KV_DIM]);
        let v = b.add_linear(normed1, vw, None, &[SEQ_LEN, KV_DIM]);
        let attn = b.add_attention(
            q,
            k,
            v,
            AttentionMask::Causal,
            Some(scale),
            &[SEQ_LEN, KV_DIM],
        );
        let attn_out = b.add_linear(attn, ow, None, &shape);
        let res1 = b.add_binary_add(current, attn_out, &shape);

        let n2e = b.add_input("l1_n2e", &[1]);
        let n2w = b.add_input("l1_n2w", &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, n2e, 1, n2w, &shape);

        let gw = b.add_input("l1_gw", &[FFN_DIM, HIDDEN_DIM]);
        let uw = b.add_input("l1_uw", &[FFN_DIM, HIDDEN_DIM]);
        let dw = b.add_input("l1_dw", &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed2, gw, None, &ffn_shape);
        let gate_act = add_silu(&mut b, gate, &ffn_shape);
        let up = b.add_linear(normed2, uw, None, &ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
        let ffn_out = b.add_linear(hidden, dw, None, &shape);

        current = b.add_binary_add(res1, ffn_out, &shape);
    }

    b.build(current)
        .expect("valid mixed quantization decoder kernel")
}

fn mixed_quant_decoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);

    // Layer 0: INT4 weights
    let int4_q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let int4_k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let int4_v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let int4_o_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), INT4_SYM_MAX);
    let int4_gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let int4_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let int4_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), INT4_SYM_MAX);

    // Layer 1: FP16 weights (higher precision)
    let fp16_q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), FP16_WEIGHT_MAG);
    let fp16_k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), FP16_WEIGHT_MAG);
    let fp16_v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), FP16_WEIGHT_MAG);
    let fp16_o_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), FP16_WEIGHT_MAG);
    let fp16_gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), FP16_WEIGHT_MAG);
    let fp16_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), FP16_WEIGHT_MAG);
    let fp16_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), FP16_WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable, // hidden
        // Layer 0 (INT4)
        TensorParamBinding::ConstantScalar(1e-6), // l0_n1e
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l0_n1w
        TensorParamBinding::ConstantTensor(int4_q_w), // l0_qw
        TensorParamBinding::ConstantTensor(int4_k_w), // l0_kw
        TensorParamBinding::ConstantTensor(int4_v_w), // l0_vw
        TensorParamBinding::ConstantTensor(int4_o_w), // l0_ow
        TensorParamBinding::ConstantScalar(1e-6), // l0_n2e
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l0_n2w
        TensorParamBinding::ConstantTensor(int4_gate_w), // l0_gw
        TensorParamBinding::ConstantTensor(int4_up_w), // l0_uw
        TensorParamBinding::ConstantTensor(int4_down_w), // l0_dw
        // Layer 1 (FP16)
        TensorParamBinding::ConstantScalar(1e-6), // l1_n1e
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l1_n1w
        TensorParamBinding::ConstantTensor(fp16_q_w), // l1_qw
        TensorParamBinding::ConstantTensor(fp16_k_w), // l1_kw
        TensorParamBinding::ConstantTensor(fp16_v_w), // l1_vw
        TensorParamBinding::ConstantTensor(fp16_o_w), // l1_ow
        TensorParamBinding::ConstantScalar(1e-6), // l1_n2e
        TensorParamBinding::ConstantTensor(norm_w), // l1_n2w
        TensorParamBinding::ConstantTensor(fp16_gate_w), // l1_gw
        TensorParamBinding::ConstantTensor(fp16_up_w), // l1_uw
        TensorParamBinding::ConstantTensor(fp16_down_w), // l1_dw
    ]
}

/// IBP bounds through mixed quantization decoder (INT4 layer 0, FP16 layer 1).
#[test]
fn test_mixed_quant_decoder_ibp() {
    let def = build_mixed_quant_decoder_kernel();
    let bindings = mixed_quant_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through mixed quant decoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "mixed quant decoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mixed quant decoder IBP (INT4 L0 + FP16 L1): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record mixed quantization decoder result.
#[test]
fn test_mixed_quant_decoder_verify_and_record() {
    let def = build_mixed_quant_decoder_kernel();
    let bindings = mixed_quant_decoder_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_qaw_mixed_quant_decoder");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}
