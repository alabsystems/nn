// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: GLM-OCR subgraph NY composition.
//!
//! Verifies bounds propagation through GLM-OCR sub-blocks used in the
//! dpdf document understanding pipeline for optical character recognition:
//!
//! 1. **RMSNorm IBP/CROWN**: Root mean square normalization used in
//!    GLM-4V decoder layers. Verifies bounded output given bounded input.
//!
//! 2. **SwiGLU FFN**: gate_proj -> SiLU -> mul(up_proj) -> down_proj
//!    Standard GLM FFN with gated linear units.
//!
//! 3. **GQA attention + softmax**: Q/K/V projection with grouped-query
//!    attention and softmax. Output bounded.
//!
//! 4. **Rotary embedding bounds**: cos/sin positional encoding bounded
//!    within [-1, 1].
//!
//! 5. **Decoder layer**: Attention -> RMSNorm -> SwiGLU -> RMSNorm.
//!    Full decoder layer composition.
//!
//! 6. **MTP head**: Linear -> softmax. Output distribution in [0, 1].
//!
//! 7. **MTP multi-step**: Chain of MTP heads consuming prior hidden state.
//!
//! 8. **Embedding projection**: Token embedding -> Linear projection bounds.
//!
//! 9. **Causal mask attention**: Causal mask preserves attention bounds.
//!
//! 10. **Full decoder stack**: 2-layer decoder -> LM head. End-to-end.
//!
//! Architecture references:
//! - GLM-4V (THUDM): Vision-language model with GLM-4 decoder
//! - RMSNorm (Zhang & Sennrich, 2019): replaces LayerNorm in GLM
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN
//! - GQA (Ainslie et al., 2023): Grouped-Query Attention
//!
//! Dimensions (small for fast verification):
//! - HIDDEN_DIM=64, FFN_DIM=128, SEQ_LEN=4, NUM_HEADS=4, NUM_KV_HEADS=2
//!
//! Part of #3884: NY compose tests for GLM-OCR subgraphs.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Hidden dimension (GLM decoder hidden size, tiny for testing).
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
/// Vocabulary size for embedding/LM head tests.
const VOCAB_SIZE: usize = 256;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// 1. RMSNorm IBP
// ===========================================================================

/// Build an RMSNorm kernel for the GLM-OCR decoder.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, hidden states in [-1, 1]).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_glm_ocr_rmsnorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_rmsnorm");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[HIDDEN_DIM]);

    let out = b.add_rms_norm(input, eps, 1, weight, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid GLM-OCR RMSNorm kernel")
}

/// Bindings for RMSNorm.
fn glm_ocr_rmsnorm_bindings() -> Vec<TensorParamBinding> {
    let weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(weight),
    ]
}

/// RMSNorm IBP bounds propagate with [-1, 1] hidden states.
#[test]
fn test_rms_norm_ibp() {
    let def = build_glm_ocr_rmsnorm_kernel();
    let bindings = glm_ocr_rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR RMSNorm");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "RMSNorm output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR RMSNorm IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. RMSNorm CROWN
// ===========================================================================

/// CROWN bounds propagate through RMSNorm.
///
/// RMSNorm involves division by sqrt(mean(x^2) + eps), which requires
/// CROWN linearization. Uses IbpValidated mode per nn engineering rules.
#[test]
fn test_rms_norm_crown() {
    let def = build_glm_ocr_rmsnorm_kernel();
    let bindings = glm_ocr_rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR RMSNorm CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record RMSNorm.
#[test]
fn test_rms_norm_verify_and_record() {
    let def = build_glm_ocr_rmsnorm_kernel();
    let bindings = glm_ocr_rmsnorm_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "glm_ocr_rmsnorm");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 3. SwiGLU FFN with CROWN
// ===========================================================================

/// Build a SwiGLU FFN kernel (GLM-4V style).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// SwiGLU FFN architecture:
///   gate = gate_proj(x)                    [SEQ_LEN, FFN_DIM]
///   gate_activated = silu(gate)            [SEQ_LEN, FFN_DIM]
///   up = up_proj(x)                        [SEQ_LEN, FFN_DIM]
///   hidden = gate_activated * up           [SEQ_LEN, FFN_DIM]
///   output = down_proj(hidden)             [SEQ_LEN, HIDDEN_DIM]
fn build_glm_ocr_swiglu_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_swiglu_ffn");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let gate_w = b.add_input("gate_proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_proj_weight", &[HIDDEN_DIM, FFN_DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // Gate branch: gate_proj -> SiLU
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    // SiLU(x) = x * sigmoid(x)
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);

    // Up branch: up_proj
    let up = b.add_linear(input, up_w, None, &ffn_shape);

    // Multiplicative gating
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);

    // Down projection
    let out = b.add_linear(hidden, down_w, None, &out_shape);

    b.build(out).expect("valid GLM-OCR SwiGLU FFN kernel")
}

/// Bindings for SwiGLU FFN.
fn glm_ocr_swiglu_ffn_bindings() -> Vec<TensorParamBinding> {
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(gate_w),
        TensorParamBinding::ConstantTensor(up_w),
        TensorParamBinding::ConstantTensor(down_w),
    ]
}

/// CROWN bounds propagate through SwiGLU FFN.
///
/// Sigmoid in SiLU is piecewise-smooth and CROWN-friendly. Multiplicative
/// gating uses McCormick envelope relaxation.
#[test]
fn test_swiglu_ffn_crown() {
    let def = build_glm_ocr_swiglu_ffn_kernel();
    let bindings = glm_ocr_swiglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR SwiGLU FFN CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record SwiGLU FFN.
#[test]
fn test_swiglu_ffn_verify_and_record() {
    let def = build_glm_ocr_swiglu_ffn_kernel();
    let bindings = glm_ocr_swiglu_ffn_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "glm_ocr_swiglu_ffn");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 4. GQA attention + softmax IBP
// ===========================================================================

/// Build a grouped-query attention kernel with softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// GQA with NUM_HEADS=4 query heads and NUM_KV_HEADS=2 KV heads.
/// Each KV head is shared by 2 query heads. For verification we project
/// Q to [SEQ_LEN, HIDDEN_DIM] and K/V to [SEQ_LEN, KV_DIM], then use
/// the monolithic attention op which handles softmax internally.
fn build_glm_ocr_gqa_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_gqa_attention");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_proj_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_proj_weight", &[KV_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Q projection: [SEQ_LEN, HIDDEN_DIM] -> [SEQ_LEN, HIDDEN_DIM]
    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // K/V projections: [SEQ_LEN, HIDDEN_DIM] -> [SEQ_LEN, KV_DIM]
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, KV_DIM]);

    // For GQA: repeat KV heads to match Q heads. Reshape K/V from
    // [SEQ_LEN, KV_DIM] -> [SEQ_LEN, NUM_KV_HEADS, HEAD_DIM] then
    // broadcast to [SEQ_LEN, NUM_HEADS, HEAD_DIM] -> reshape back.
    // Simplified: use KV_DIM projection and flatten for attention.
    // For tractability, use standard attention on matching-dim Q/K/V:
    // Reshape Q to [SEQ_LEN, NUM_KV_HEADS, HEAD_DIM * (NUM_HEADS/NUM_KV_HEADS)]
    // is complex, so we verify Q/K/V -> softmax attention with matching dims.
    //
    // Instead: project Q down to KV_DIM for attention, then project back up.
    let q_down_w = b.add_input("q_down_weight", &[KV_DIM, HIDDEN_DIM]);
    let q_down = b.add_linear(input, q_down_w, None, &[SEQ_LEN, KV_DIM]);

    // Attention: Q_down @ K^T -> softmax -> @ V -> [SEQ_LEN, KV_DIM]
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn_out = b.add_attention(
        q_down,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );

    // Output projection: [SEQ_LEN, KV_DIM] -> [SEQ_LEN, HIDDEN_DIM]
    let out_up_w = b.add_input("out_up_weight", &[HIDDEN_DIM, KV_DIM]);
    let out = b.add_linear(attn_out, out_up_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Residual connection
    let _ = q; // Q was projected but not used in simplified path; avoid warning
    let _ = out_w; // out_w unused in simplified path
    let result = b.add_binary_add(input, out, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(result).expect("valid GLM-OCR GQA attention kernel")
}

/// Bindings for GQA attention.
fn glm_ocr_gqa_attention_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let q_down_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_up_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                 // hidden
        TensorParamBinding::ConstantTensor(q_w),      // q_proj_weight
        TensorParamBinding::ConstantTensor(k_w),      // k_proj_weight
        TensorParamBinding::ConstantTensor(v_w),      // v_proj_weight
        TensorParamBinding::ConstantTensor(out_w),    // out_proj_weight
        TensorParamBinding::ConstantTensor(q_down_w), // q_down_weight
        TensorParamBinding::ConstantTensor(out_up_w), // out_up_weight
    ]
}

/// IBP bounds propagate through GQA attention with softmax.
#[test]
fn test_gqa_attention_softmax_ibp() {
    let def = build_glm_ocr_gqa_attention_kernel();
    let bindings = glm_ocr_gqa_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR GQA attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "GQA attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR GQA attention IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Residual connection preserves bounded output
    assert!(
        lo_min > -100.0,
        "GQA attention lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 5. Rotary embedding bounds
// ===========================================================================

/// Build a rotary embedding bounds verification kernel.
///
/// Rotary embeddings use cos/sin, which are bounded in [-1, 1].
/// We build a simplified RoPE: compute cos and sin of position-scaled
/// input, then combine via the rotation formula.
///
/// Input: `[SEQ_LEN, HEAD_DIM]` (Variable, query/key vectors).
/// Output: `[SEQ_LEN, HEAD_DIM]`.
///
/// RoPE: x_rot[2i] = x[2i]*cos(theta) - x[2i+1]*sin(theta)
///        x_rot[2i+1] = x[2i]*sin(theta) + x[2i+1]*cos(theta)
///
/// For verification tractability, we model this as:
///   cos_pe, sin_pe are constant tensors with values in [-1, 1]
///   output = x * cos_pe + rotate(x) * sin_pe
/// where rotate swaps pairs. We approximate with element-wise multiply.
fn build_glm_ocr_rotary_embedding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_rotary_embedding");

    let input = b.add_input("qk", &[SEQ_LEN, HEAD_DIM]);
    let cos_pe = b.add_input("cos_pe", &[SEQ_LEN, HEAD_DIM]);
    let sin_pe = b.add_input("sin_pe", &[SEQ_LEN, HEAD_DIM]);

    let shape = [SEQ_LEN, HEAD_DIM];

    // x * cos(theta)
    let x_cos = b.add_binary_mul(input, cos_pe, &shape);

    // For the sin term, we need the rotated input. In full RoPE this
    // swaps even/odd pairs and negates. For verification we use the
    // input directly (conservative: over-approximates the actual rotation).
    // x * sin(theta) -- represents the rotated contribution
    let x_sin = b.add_binary_mul(input, sin_pe, &shape);

    // output = x*cos + rotated_x*sin (simplified as x*cos + x*sin)
    let out = b.add_binary_add(x_cos, x_sin, &shape);

    b.build(out).expect("valid GLM-OCR rotary embedding kernel")
}

/// Bindings for rotary embedding.
///
/// cos_pe and sin_pe are constant with values in [-1, 1] (the range
/// of cosine and sine functions).
fn glm_ocr_rotary_embedding_bindings() -> Vec<TensorParamBinding> {
    // Generate representative cos/sin values
    let n = SEQ_LEN * HEAD_DIM;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for t in 0..SEQ_LEN {
        for d in 0..HEAD_DIM {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * (d / 2) as f64 / HEAD_DIM as f64);
            cos_data.push(freq.cos() as f32);
            sin_data.push(freq.sin() as f32);
        }
    }
    let cos_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HEAD_DIM]), cos_data).expect("valid cos shape");
    let sin_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HEAD_DIM]), sin_data).expect("valid sin shape");

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_pe),
        TensorParamBinding::ConstantTensor(sin_pe),
    ]
}

/// Rotary embedding bounds: cos/sin constants are in [-1, 1],
/// so output bounds scale linearly with input bounds.
#[test]
fn test_rotary_embedding_bounds() {
    let def = build_glm_ocr_rotary_embedding_kernel();
    let bindings = glm_ocr_rotary_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HEAD_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR rotary embedding");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HEAD_DIM],
        "rotary embedding output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR rotary embedding IBP (qk [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // output = x*cos + x*sin, with x in [-1,1] and cos/sin in [-1,1]
    // each product is in [-1,1], sum is in [-2, 2]
    assert!(
        hi_max <= 2.0 + 1e-4,
        "rotary embedding output should be <= 2 with unit input, got {hi_max}"
    );
    assert!(
        lo_min >= -2.0 - 1e-4,
        "rotary embedding output should be >= -2 with unit input, got {lo_min}"
    );
}

// ===========================================================================
// 6. Decoder layer compose CROWN
// ===========================================================================

/// Build a full GLM-OCR decoder layer:
/// RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_glm_ocr_decoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_decoder_layer");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // Self-attention (simplified: Q/K/V projection + attention + output projection)
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Residual connection after attention
    let residual1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(residual1, norm2_eps, 1, norm2_w, &shape);

    // SwiGLU FFN
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual connection after FFN
    let out = b.add_binary_add(residual1, ffn_out, &shape);

    b.build(out).expect("valid GLM-OCR decoder layer kernel")
}

/// Bindings for full decoder layer.
fn glm_ocr_decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(q_w),            // q_weight
        TensorParamBinding::ConstantTensor(k_w),            // k_weight
        TensorParamBinding::ConstantTensor(v_w),            // v_weight
        TensorParamBinding::ConstantTensor(out_w),          // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
    ]
}

/// CROWN bounds propagate through full decoder layer.
#[test]
fn test_decoder_layer_compose_crown() {
    let def = build_glm_ocr_decoder_layer_kernel();
    let bindings = glm_ocr_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR decoder layer: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record decoder layer.
#[test]
fn test_decoder_layer_verify_and_record() {
    let def = build_glm_ocr_decoder_layer_kernel();
    let bindings = glm_ocr_decoder_layer_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "glm_ocr_decoder_layer");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 7. MTP head: Linear -> softmax (output in [0, 1])
// ===========================================================================

/// Build an MTP (multi-token prediction) head kernel.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution).
///
/// Architecture: Linear(HIDDEN_DIM -> VOCAB_SIZE) -> softmax(dim=-1)
fn build_glm_ocr_mtp_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_mtp_head");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    // Linear projection to vocabulary
    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax over vocabulary dimension
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid GLM-OCR MTP head kernel")
}

/// Bindings for MTP head.
fn glm_ocr_mtp_head_bindings() -> Vec<TensorParamBinding> {
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(lm_w),
    ]
}

/// IBP bounds propagate through MTP head.
///
/// Softmax output is a probability distribution: all elements in [0, 1].
#[test]
fn test_mtp_head_softmax_ibp() {
    let def = build_glm_ocr_mtp_head_kernel();
    let bindings = glm_ocr_mtp_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR MTP head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "MTP head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR MTP head IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    // Softmax codomain is (0, 1).
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
// 8. MTP multi-step IBP
// ===========================================================================

/// Build a multi-step MTP chain: two sequential linear projections,
/// each followed by softmax, modeling multi-token prediction heads
/// that consume the prior step's hidden state.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (final step distribution).
///
/// Step 1: Linear -> softmax -> [SEQ_LEN, VOCAB_SIZE]
/// Step 2: Linear(VOCAB_SIZE -> HIDDEN_DIM) -> Linear(HIDDEN_DIM -> VOCAB_SIZE) -> softmax
fn build_glm_ocr_mtp_multi_step_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_mtp_multi_step");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w1 = b.add_input("lm_head_weight_1", &[VOCAB_SIZE, HIDDEN_DIM]);
    let down_w = b.add_input("down_proj_weight", &[HIDDEN_DIM, VOCAB_SIZE]);
    let lm_w2 = b.add_input("lm_head_weight_2", &[VOCAB_SIZE, HIDDEN_DIM]);

    // Step 1: project to vocab, softmax
    let logits1 = b.add_linear(input, lm_w1, None, &[SEQ_LEN, VOCAB_SIZE]);
    let _probs1 = b.add_softmax(logits1, 1, &[SEQ_LEN, VOCAB_SIZE]);

    // Step 2: project softmax output back to hidden, then to vocab again
    // Use logits (pre-softmax) as hidden state for step 2 (more numerically
    // interesting for verification than post-softmax which is in [0,1])
    let hidden2 = b.add_linear(logits1, down_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let logits2 = b.add_linear(hidden2, lm_w2, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs2 = b.add_softmax(logits2, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs2)
        .expect("valid GLM-OCR MTP multi-step kernel")
}

/// Bindings for MTP multi-step.
fn glm_ocr_mtp_multi_step_bindings() -> Vec<TensorParamBinding> {
    let lm_w1 = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, VOCAB_SIZE]), WEIGHT_MAG);
    let lm_w2 = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(lm_w1),
        TensorParamBinding::ConstantTensor(down_w),
        TensorParamBinding::ConstantTensor(lm_w2),
    ]
}

/// IBP bounds through multi-step MTP chain.
///
/// Both steps produce softmax output in [0, 1]. The chain should
/// maintain bounded output through composition.
#[test]
fn test_mtp_multi_step_ibp() {
    let def = build_glm_ocr_mtp_multi_step_kernel();
    let bindings = glm_ocr_mtp_multi_step_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR MTP multi-step");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "MTP multi-step output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR MTP multi-step IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    // Final softmax output must be in [0, 1]
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "multi-step softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "multi-step softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 9. Embedding projection IBP
// ===========================================================================

/// Build an embedding -> linear projection kernel.
///
/// Input: `[SEQ_LEN]` (Variable, token indices -- bounded integers).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Embedding lookup selects rows from a weight table. The embedding
/// weights are bounded constants, so the output bounds are determined
/// by the weight range. A subsequent linear projection maps to the
/// hidden dimension.
fn build_glm_ocr_embedding_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_embedding_projection");

    let input = b.add_input("token_ids", &[SEQ_LEN]);
    let emb_w = b.add_input("embedding_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Embedding lookup: [SEQ_LEN] -> [SEQ_LEN, HIDDEN_DIM]
    let embedded = b.add_embedding(input, emb_w, &[SEQ_LEN, HIDDEN_DIM]);

    // Linear projection: [SEQ_LEN, HIDDEN_DIM] -> [SEQ_LEN, HIDDEN_DIM]
    let out = b.add_linear(embedded, proj_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out)
        .expect("valid GLM-OCR embedding projection kernel")
}

/// Bindings for embedding projection.
fn glm_ocr_embedding_projection_bindings() -> Vec<TensorParamBinding> {
    let emb_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(emb_w),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

/// IBP bounds through embedding -> linear projection.
///
/// Embedding output is bounded by the weight table values. With uniform
/// 0.02 weights, embedding output is in [-0.02, 0.02] per element.
/// Linear projection scales by weight * hidden_dim.
#[test]
fn test_embedding_projection_ibp() {
    let def = build_glm_ocr_embedding_projection_kernel();
    let bindings = glm_ocr_embedding_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Token indices as bounded variable input: indices in [0, VOCAB_SIZE-1]
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), (VOCAB_SIZE - 1) as f32),
    )
    .expect("valid index bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR embedding projection");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "embedding projection output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR embedding projection IBP (indices [0,{}]): bounds=[{lo_min}, {hi_max}]",
        VOCAB_SIZE - 1
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. Causal mask attention IBP
// ===========================================================================

/// Build a causal-masked attention kernel.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Causal attention ensures position j attends only to positions <= j.
/// The softmax over masked positions should still produce valid [0, 1]
/// probabilities. Output bounds should be preserved by the mask.
fn build_glm_ocr_causal_mask_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_causal_mask_attention");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let shape = [SEQ_LEN, HIDDEN_DIM];

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);

    let scale = 1.0 / (HIDDEN_DIM as f32).sqrt();
    let out = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);

    b.build(out)
        .expect("valid GLM-OCR causal mask attention kernel")
}

/// Bindings for causal mask attention.
fn glm_ocr_causal_mask_attention_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(q_w),
        TensorParamBinding::ConstantTensor(k_w),
        TensorParamBinding::ConstantTensor(v_w),
    ]
}

/// IBP bounds propagate through causal-masked attention.
///
/// Causal mask restricts attention to past positions. The softmax
/// still produces valid probabilities, and the weighted sum of V
/// should produce bounded output.
#[test]
fn test_causal_mask_attention_ibp() {
    let def = build_glm_ocr_causal_mask_attention_kernel();
    let bindings = glm_ocr_causal_mask_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR causal mask attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "causal mask attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR causal mask attention IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through causal mask attention.
#[test]
fn test_causal_mask_attention_crown() {
    let def = build_glm_ocr_causal_mask_attention_kernel();
    let bindings = glm_ocr_causal_mask_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR causal mask attention CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 11. Full decoder stack IBP: 2-layer decoder -> LM head
// ===========================================================================

/// Build a 2-layer decoder stack with LM head.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (logit distribution via softmax).
///
/// Layer 1: RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual
/// Layer 2: (same structure)
/// LM head: RMSNorm -> Linear -> softmax
fn build_glm_ocr_full_decoder_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_full_decoder_stack");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // --- Layer 1 ---
    let norm1a_eps = b.add_input("l1_norm1_eps", &[1]);
    let norm1a_w = b.add_input("l1_norm1_weight", &[HIDDEN_DIM]);
    let normed1a = b.add_rms_norm(input, norm1a_eps, 1, norm1a_w, &shape);

    let l1_q_w = b.add_input("l1_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l1_k_w = b.add_input("l1_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l1_v_w = b.add_input("l1_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l1_out_w = b.add_input("l1_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q1 = b.add_linear(normed1a, l1_q_w, None, &shape);
    let k1 = b.add_linear(normed1a, l1_k_w, None, &shape);
    let v1 = b.add_linear(normed1a, l1_v_w, None, &shape);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn1 = b.add_attention(q1, k1, v1, AttentionMask::Causal, Some(scale), &shape);
    let attn1_out = b.add_linear(attn1, l1_out_w, None, &shape);
    let res1a = b.add_binary_add(input, attn1_out, &shape);

    let norm1b_eps = b.add_input("l1_norm2_eps", &[1]);
    let norm1b_w = b.add_input("l1_norm2_weight", &[HIDDEN_DIM]);
    let normed1b = b.add_rms_norm(res1a, norm1b_eps, 1, norm1b_w, &shape);

    let l1_gate_w = b.add_input("l1_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l1_up_w = b.add_input("l1_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l1_down_w = b.add_input("l1_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate1 = b.add_linear(normed1b, l1_gate_w, None, &ffn_shape);
    let gate1_sig = b.add_sigmoid(gate1, &ffn_shape);
    let gate1_act = b.add_binary_mul(gate1, gate1_sig, &ffn_shape);
    let up1 = b.add_linear(normed1b, l1_up_w, None, &ffn_shape);
    let hidden1 = b.add_binary_mul(gate1_act, up1, &ffn_shape);
    let ffn1_out = b.add_linear(hidden1, l1_down_w, None, &shape);
    let res1b = b.add_binary_add(res1a, ffn1_out, &shape);

    // --- Layer 2 ---
    let norm2a_eps = b.add_input("l2_norm1_eps", &[1]);
    let norm2a_w = b.add_input("l2_norm1_weight", &[HIDDEN_DIM]);
    let normed2a = b.add_rms_norm(res1b, norm2a_eps, 1, norm2a_w, &shape);

    let l2_q_w = b.add_input("l2_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l2_k_w = b.add_input("l2_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l2_v_w = b.add_input("l2_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l2_out_w = b.add_input("l2_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q2 = b.add_linear(normed2a, l2_q_w, None, &shape);
    let k2 = b.add_linear(normed2a, l2_k_w, None, &shape);
    let v2 = b.add_linear(normed2a, l2_v_w, None, &shape);
    let attn2 = b.add_attention(q2, k2, v2, AttentionMask::Causal, Some(scale), &shape);
    let attn2_out = b.add_linear(attn2, l2_out_w, None, &shape);
    let res2a = b.add_binary_add(res1b, attn2_out, &shape);

    let norm2b_eps = b.add_input("l2_norm2_eps", &[1]);
    let norm2b_w = b.add_input("l2_norm2_weight", &[HIDDEN_DIM]);
    let normed2b = b.add_rms_norm(res2a, norm2b_eps, 1, norm2b_w, &shape);

    let l2_gate_w = b.add_input("l2_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l2_up_w = b.add_input("l2_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l2_down_w = b.add_input("l2_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate2 = b.add_linear(normed2b, l2_gate_w, None, &ffn_shape);
    let gate2_sig = b.add_sigmoid(gate2, &ffn_shape);
    let gate2_act = b.add_binary_mul(gate2, gate2_sig, &ffn_shape);
    let up2 = b.add_linear(normed2b, l2_up_w, None, &ffn_shape);
    let hidden2 = b.add_binary_mul(gate2_act, up2, &ffn_shape);
    let ffn2_out = b.add_linear(hidden2, l2_down_w, None, &shape);
    let res2b = b.add_binary_add(res2a, ffn2_out, &shape);

    // --- LM Head: RMSNorm -> Linear -> softmax ---
    let final_norm_eps = b.add_input("final_norm_eps", &[1]);
    let final_norm_w = b.add_input("final_norm_weight", &[HIDDEN_DIM]);
    let final_normed = b.add_rms_norm(res2b, final_norm_eps, 1, final_norm_w, &shape);

    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(final_normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid GLM-OCR full decoder stack kernel")
}

/// Bindings for full decoder stack.
fn glm_ocr_full_decoder_stack_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable, // hidden
        // Layer 1
        TensorParamBinding::ConstantScalar(1e-5), // l1_norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l1_norm1_weight
        TensorParamBinding::ConstantTensor(q_w.clone()), // l1_q_weight
        TensorParamBinding::ConstantTensor(k_w.clone()), // l1_k_weight
        TensorParamBinding::ConstantTensor(v_w.clone()), // l1_v_weight
        TensorParamBinding::ConstantTensor(out_w.clone()), // l1_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // l1_norm2_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l1_norm2_weight
        TensorParamBinding::ConstantTensor(gate_w.clone()), // l1_gate_weight
        TensorParamBinding::ConstantTensor(up_w.clone()), // l1_up_weight
        TensorParamBinding::ConstantTensor(down_w.clone()), // l1_down_weight
        // Layer 2
        TensorParamBinding::ConstantScalar(1e-5), // l2_norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l2_norm1_weight
        TensorParamBinding::ConstantTensor(q_w),  // l2_q_weight
        TensorParamBinding::ConstantTensor(k_w),  // l2_k_weight
        TensorParamBinding::ConstantTensor(v_w),  // l2_v_weight
        TensorParamBinding::ConstantTensor(out_w), // l2_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // l2_norm2_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l2_norm2_weight
        TensorParamBinding::ConstantTensor(gate_w), // l2_gate_weight
        TensorParamBinding::ConstantTensor(up_w), // l2_up_weight
        TensorParamBinding::ConstantTensor(down_w), // l2_down_weight
        // LM Head
        TensorParamBinding::ConstantScalar(1e-5), // final_norm_eps
        TensorParamBinding::ConstantTensor(norm_w), // final_norm_weight
        TensorParamBinding::ConstantTensor(lm_w), // lm_head_weight
    ]
}

/// IBP bounds propagate through full 2-layer decoder stack with LM head.
///
/// End-to-end verification: input hidden states -> 2 decoder layers
/// -> RMSNorm -> LM head -> softmax. Output must be a valid probability
/// distribution in [0, 1].
#[test]
fn test_full_decoder_stack_ibp() {
    let def = build_glm_ocr_full_decoder_stack_kernel();
    let bindings = glm_ocr_full_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR full decoder stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "full decoder stack output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR full decoder stack IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Final softmax: output in [0, 1]
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "full decoder stack lower must be >= 0 (softmax output), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "full decoder stack upper must be <= 1 (softmax output), got {hi_max}"
    );
}

/// Verify and record full decoder stack.
#[test]
fn test_full_decoder_stack_verify_and_record() {
    let def = build_glm_ocr_full_decoder_stack_kernel();
    let bindings = glm_ocr_full_decoder_stack_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "glm_ocr_full_decoder_stack");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ===========================================================================
// 12. Two-layer decoder stack IBP (deeper composition)
// ===========================================================================

/// Build a 2-layer decoder stack WITHOUT LM head for intermediate
/// hidden-state verification.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Each layer: RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
fn build_glm_ocr_two_layer_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_two_layer_decoder");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // --- Layer 1 ---
    let l1n1_eps = b.add_input("l1_norm1_eps", &[1]);
    let l1n1_w = b.add_input("l1_norm1_weight", &[HIDDEN_DIM]);
    let normed1a = b.add_rms_norm(input, l1n1_eps, 1, l1n1_w, &shape);

    let l1_q_w = b.add_input("l1_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l1_k_w = b.add_input("l1_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l1_v_w = b.add_input("l1_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l1_out_w = b.add_input("l1_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q1 = b.add_linear(normed1a, l1_q_w, None, &shape);
    let k1 = b.add_linear(normed1a, l1_k_w, None, &shape);
    let v1 = b.add_linear(normed1a, l1_v_w, None, &shape);
    let attn1 = b.add_attention(q1, k1, v1, AttentionMask::Causal, Some(scale), &shape);
    let attn1_out = b.add_linear(attn1, l1_out_w, None, &shape);
    let res1a = b.add_binary_add(input, attn1_out, &shape);

    let l1n2_eps = b.add_input("l1_norm2_eps", &[1]);
    let l1n2_w = b.add_input("l1_norm2_weight", &[HIDDEN_DIM]);
    let normed1b = b.add_rms_norm(res1a, l1n2_eps, 1, l1n2_w, &shape);

    let l1_gate_w = b.add_input("l1_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l1_up_w = b.add_input("l1_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l1_down_w = b.add_input("l1_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate1 = b.add_linear(normed1b, l1_gate_w, None, &ffn_shape);
    let gate1_sig = b.add_sigmoid(gate1, &ffn_shape);
    let gate1_act = b.add_binary_mul(gate1, gate1_sig, &ffn_shape);
    let up1 = b.add_linear(normed1b, l1_up_w, None, &ffn_shape);
    let hidden1 = b.add_binary_mul(gate1_act, up1, &ffn_shape);
    let ffn1_out = b.add_linear(hidden1, l1_down_w, None, &shape);
    let res1b = b.add_binary_add(res1a, ffn1_out, &shape);

    // --- Layer 2 ---
    let l2n1_eps = b.add_input("l2_norm1_eps", &[1]);
    let l2n1_w = b.add_input("l2_norm1_weight", &[HIDDEN_DIM]);
    let normed2a = b.add_rms_norm(res1b, l2n1_eps, 1, l2n1_w, &shape);

    let l2_q_w = b.add_input("l2_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l2_k_w = b.add_input("l2_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l2_v_w = b.add_input("l2_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l2_out_w = b.add_input("l2_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q2 = b.add_linear(normed2a, l2_q_w, None, &shape);
    let k2 = b.add_linear(normed2a, l2_k_w, None, &shape);
    let v2 = b.add_linear(normed2a, l2_v_w, None, &shape);
    let attn2 = b.add_attention(q2, k2, v2, AttentionMask::Causal, Some(scale), &shape);
    let attn2_out = b.add_linear(attn2, l2_out_w, None, &shape);
    let res2a = b.add_binary_add(res1b, attn2_out, &shape);

    let l2n2_eps = b.add_input("l2_norm2_eps", &[1]);
    let l2n2_w = b.add_input("l2_norm2_weight", &[HIDDEN_DIM]);
    let normed2b = b.add_rms_norm(res2a, l2n2_eps, 1, l2n2_w, &shape);

    let l2_gate_w = b.add_input("l2_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l2_up_w = b.add_input("l2_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l2_down_w = b.add_input("l2_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate2 = b.add_linear(normed2b, l2_gate_w, None, &ffn_shape);
    let gate2_sig = b.add_sigmoid(gate2, &ffn_shape);
    let gate2_act = b.add_binary_mul(gate2, gate2_sig, &ffn_shape);
    let up2 = b.add_linear(normed2b, l2_up_w, None, &ffn_shape);
    let hidden2 = b.add_binary_mul(gate2_act, up2, &ffn_shape);
    let ffn2_out = b.add_linear(hidden2, l2_down_w, None, &shape);
    let out = b.add_binary_add(res2a, ffn2_out, &shape);

    b.build(out)
        .expect("valid GLM-OCR two-layer decoder kernel")
}

/// Bindings for two-layer decoder stack (no LM head).
fn glm_ocr_two_layer_decoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable, // hidden
        // Layer 1
        TensorParamBinding::ConstantScalar(1e-5), // l1_norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l1_norm1_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // l1_q_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // l1_k_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // l1_v_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // l1_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // l1_norm2_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l1_norm2_weight
        TensorParamBinding::ConstantTensor(gate_w.clone()), // l1_gate_weight
        TensorParamBinding::ConstantTensor(up_w.clone()), // l1_up_weight
        TensorParamBinding::ConstantTensor(down_w.clone()), // l1_down_weight
        // Layer 2
        TensorParamBinding::ConstantScalar(1e-5), // l2_norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l2_norm1_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // l2_q_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // l2_k_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // l2_v_weight
        TensorParamBinding::ConstantTensor(qkv_w), // l2_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // l2_norm2_eps
        TensorParamBinding::ConstantTensor(norm_w), // l2_norm2_weight
        TensorParamBinding::ConstantTensor(gate_w), // l2_gate_weight
        TensorParamBinding::ConstantTensor(up_w), // l2_up_weight
        TensorParamBinding::ConstantTensor(down_w), // l2_down_weight
    ]
}

/// IBP bounds through 2-layer decoder (no LM head).
///
/// Verifies residual connections preserve bounded hidden states across
/// two chained decoder layers with RMSNorm + Attention + SwiGLU each.
#[test]
fn test_decoder_two_layer_stack_ibp() {
    let def = build_glm_ocr_two_layer_decoder_kernel();
    let bindings = glm_ocr_two_layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR two-layer decoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "two-layer decoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR two-layer decoder IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 13. Two-layer decoder stack CROWN
// ===========================================================================

/// CROWN bounds through 2-layer decoder stack.
///
/// Tests CROWN linearization depth: two RMSNorm layers + two attention
/// blocks + two SwiGLU FFN blocks with McCormick envelope relaxation.
#[test]
fn test_decoder_two_layer_stack_crown() {
    let def = build_glm_ocr_two_layer_decoder_kernel();
    let bindings = glm_ocr_two_layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR two-layer decoder CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 14. Four-layer decoder stack IBP (CROWN depth stress)
// ===========================================================================

/// Build a 4-layer decoder stack for CROWN depth testing.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// 4 identical decoder layers chained. Tests bounds propagation at
/// realistic depth (GLM-4V uses 40 layers; 4 is the verification proxy).
fn build_glm_ocr_four_layer_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_four_layer_decoder");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer in 0..4 {
        let prefix = format!("l{layer}");

        // Pre-attention RMSNorm
        let n1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let n1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed_a = b.add_rms_norm(current, n1_eps, 1, n1_w, &shape);

        // Self-attention
        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed_a, q_w, None, &shape);
        let k = b.add_linear(normed_a, k_w, None, &shape);
        let v = b.add_linear(normed_a, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res_a = b.add_binary_add(current, attn_out, &shape);

        // Pre-FFN RMSNorm
        let n2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let n2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed_b = b.add_rms_norm(res_a, n2_eps, 1, n2_w, &shape);

        // SwiGLU FFN
        let gate_w = b.add_input(&format!("{prefix}_gate_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let up_w = b.add_input(&format!("{prefix}_up_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("{prefix}_down_weight"), &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed_b, gate_w, None, &ffn_shape);
        let gate_sig = b.add_sigmoid(gate, &ffn_shape);
        let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
        let up = b.add_linear(normed_b, up_w, None, &ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
        let ffn_out = b.add_linear(hidden, down_w, None, &shape);
        current = b.add_binary_add(res_a, ffn_out, &shape);
    }

    b.build(current)
        .expect("valid GLM-OCR four-layer decoder kernel")
}

/// Bindings for 4-layer decoder stack.
fn glm_ocr_four_layer_decoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gate_weight
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // up_weight
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // down_weight
    }

    bindings
}

/// IBP bounds through 4-layer decoder stack.
///
/// Stress test: 4 chained decoder layers with residual connections.
/// Verifies that bounds remain finite and non-vacuous after multiple
/// RMSNorm + Attention + SwiGLU compositions.
#[test]
fn test_decoder_four_layer_stack_ibp() {
    let def = build_glm_ocr_four_layer_decoder_kernel();
    let bindings = glm_ocr_four_layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR four-layer decoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "four-layer decoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR four-layer decoder IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 15. MTP three-head chain IBP
// ===========================================================================

/// Build a 3-step MTP prediction head chain.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (final step distribution).
///
/// Step 1: Linear(HIDDEN -> VOCAB) -> softmax
/// Step 2: Linear(VOCAB -> HIDDEN) -> Linear(HIDDEN -> VOCAB) -> softmax
/// Step 3: Linear(VOCAB -> HIDDEN) -> Linear(HIDDEN -> VOCAB) -> softmax
fn build_glm_ocr_mtp_three_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_mtp_three_head");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let vocab_shape = [SEQ_LEN, VOCAB_SIZE];
    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];

    // Step 1: project to vocab
    let lm_w1 = b.add_input("lm_head_weight_1", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits1 = b.add_linear(input, lm_w1, None, &vocab_shape);
    let _probs1 = b.add_softmax(logits1, 1, &vocab_shape);

    // Step 2: project back to hidden, then to vocab
    let down_w2 = b.add_input("down_proj_weight_2", &[HIDDEN_DIM, VOCAB_SIZE]);
    let lm_w2 = b.add_input("lm_head_weight_2", &[VOCAB_SIZE, HIDDEN_DIM]);
    let hidden2 = b.add_linear(logits1, down_w2, None, &hidden_shape);
    let logits2 = b.add_linear(hidden2, lm_w2, None, &vocab_shape);
    let _probs2 = b.add_softmax(logits2, 1, &vocab_shape);

    // Step 3: project back to hidden, then to vocab
    let down_w3 = b.add_input("down_proj_weight_3", &[HIDDEN_DIM, VOCAB_SIZE]);
    let lm_w3 = b.add_input("lm_head_weight_3", &[VOCAB_SIZE, HIDDEN_DIM]);
    let hidden3 = b.add_linear(logits2, down_w3, None, &hidden_shape);
    let logits3 = b.add_linear(hidden3, lm_w3, None, &vocab_shape);
    let probs3 = b.add_softmax(logits3, 1, &vocab_shape);

    b.build(probs3)
        .expect("valid GLM-OCR MTP three-head kernel")
}

/// Bindings for 3-step MTP chain.
fn glm_ocr_mtp_three_head_bindings() -> Vec<TensorParamBinding> {
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, VOCAB_SIZE]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(lm_w.clone()), // lm_head_weight_1
        TensorParamBinding::ConstantTensor(down_w.clone()), // down_proj_weight_2
        TensorParamBinding::ConstantTensor(lm_w.clone()), // lm_head_weight_2
        TensorParamBinding::ConstantTensor(down_w),       // down_proj_weight_3
        TensorParamBinding::ConstantTensor(lm_w),         // lm_head_weight_3
    ]
}

/// IBP bounds through 3-step MTP prediction head chain.
///
/// Three sequential Linear -> softmax heads. Each softmax constrains
/// output to [0, 1]. The chain tests that bounds remain valid through
/// multiple softmax compositions.
#[test]
fn test_mtp_three_head_chain_ibp() {
    let def = build_glm_ocr_mtp_three_head_kernel();
    let bindings = glm_ocr_mtp_three_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR MTP three-head chain");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "MTP three-head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR MTP three-head IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    // Final softmax output must be in [0, 1]
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "three-head softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "three-head softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 16. Embedding -> decoder -> LM head (end-to-end) IBP
// ===========================================================================

/// Build embedding -> single decoder layer -> LM head -> softmax.
///
/// Input: `[SEQ_LEN]` (Variable, token indices).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution).
///
/// End-to-end: embedding lookup -> decoder layer -> RMSNorm -> linear -> softmax.
fn build_glm_ocr_embedding_to_lm_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_embedding_to_lm_head");

    let input = b.add_input("token_ids", &[SEQ_LEN]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Embedding lookup
    let emb_w = b.add_input("embedding_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let embedded = b.add_embedding(input, emb_w, &shape);

    // Decoder layer: RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU -> residual
    let n1_eps = b.add_input("norm1_eps", &[1]);
    let n1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(embedded, n1_eps, 1, n1_w, &shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(embedded, attn_out, &shape);

    let n2_eps = b.add_input("norm2_eps", &[1]);
    let n2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    let res2 = b.add_binary_add(res1, ffn_out, &shape);

    // LM head: RMSNorm -> Linear -> softmax
    let fn_eps = b.add_input("final_norm_eps", &[1]);
    let fn_w = b.add_input("final_norm_weight", &[HIDDEN_DIM]);
    let final_normed = b.add_rms_norm(res2, fn_eps, 1, fn_w, &shape);

    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(final_normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid GLM-OCR embedding to LM head kernel")
}

/// Bindings for embedding -> decoder -> LM head.
fn glm_ocr_embedding_to_lm_head_bindings() -> Vec<TensorParamBinding> {
    let emb_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // token_ids
        TensorParamBinding::ConstantTensor(emb_w),          // embedding_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()),  // q_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()),  // k_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()),  // v_weight
        TensorParamBinding::ConstantTensor(qkv_w),          // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
        TensorParamBinding::ConstantScalar(1e-5),           // final_norm_eps
        TensorParamBinding::ConstantTensor(norm_w),         // final_norm_weight
        TensorParamBinding::ConstantTensor(lm_w),           // lm_head_weight
    ]
}

/// IBP end-to-end: embedding -> decoder -> LM head -> softmax.
///
/// Full pipeline from token indices to probability distribution.
/// Embedding output is bounded by weight table range, decoder preserves
/// bounds via residual connections, softmax produces [0, 1] output.
#[test]
fn test_embedding_to_lm_head_ibp() {
    let def = build_glm_ocr_embedding_to_lm_head_kernel();
    let bindings = glm_ocr_embedding_to_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Token indices bounded in [0, VOCAB_SIZE-1]
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), (VOCAB_SIZE - 1) as f32),
    )
    .expect("valid index bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR embedding to LM head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "embedding to LM head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR embedding to LM head IBP (indices [0,{}]): bounds=[{lo_min}, {hi_max}]",
        VOCAB_SIZE - 1
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Final softmax: output in [0, 1]
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "embedding to LM head lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "embedding to LM head upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 17. GQA with explicit 4:1 repeat_kv ratio IBP
// ===========================================================================

/// Build a GQA attention block with explicit 4:1 head ratio.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// NUM_HEADS=4 query heads, NUM_KV_HEADS=1 KV head (4:1 ratio).
/// Each KV head serves all 4 query heads. K/V are projected to
/// [SEQ_LEN, HEAD_DIM], then the attention operates with Q having
/// HIDDEN_DIM and K/V having HEAD_DIM.
fn build_glm_ocr_gqa_repeat_kv_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_gqa_repeat_kv");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Q projection: full HIDDEN_DIM (4 heads * HEAD_DIM)
    let q_w = b.add_input("q_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    // K/V projection: 1 KV head * HEAD_DIM = HEAD_DIM
    let k_w = b.add_input("k_proj_weight", &[HEAD_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_proj_weight", &[HEAD_DIM, HIDDEN_DIM]);

    let _q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, HEAD_DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, HEAD_DIM]);

    // For verification tractability with matching dims: project Q down
    // to HEAD_DIM to match K/V, run attention, project back up.
    let q_down_w = b.add_input("q_down_weight", &[HEAD_DIM, HIDDEN_DIM]);
    let q_down = b.add_linear(input, q_down_w, None, &[SEQ_LEN, HEAD_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn_out = b.add_attention(
        q_down,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, HEAD_DIM],
    );

    // Project back to HIDDEN_DIM
    let out_w = b.add_input("out_proj_weight", &[HIDDEN_DIM, HEAD_DIM]);
    let projected = b.add_linear(attn_out, out_w, None, &shape);

    // Residual
    let out = b.add_binary_add(input, projected, &shape);

    b.build(out).expect("valid GLM-OCR GQA repeat_kv kernel")
}

/// Bindings for GQA 4:1 repeat_kv.
fn glm_ocr_gqa_repeat_kv_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[HEAD_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[HEAD_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let q_down_w = ArrayD::from_elem(IxDyn(&[HEAD_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HEAD_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(q_w),
        TensorParamBinding::ConstantTensor(k_w),
        TensorParamBinding::ConstantTensor(v_w),
        TensorParamBinding::ConstantTensor(q_down_w),
        TensorParamBinding::ConstantTensor(out_w),
    ]
}

/// IBP bounds through GQA with 4:1 KV head ratio.
///
/// With only 1 KV head serving 4 query heads, the key/value
/// projections are smaller (HEAD_DIM vs HIDDEN_DIM), producing
/// tighter intermediate bounds. Residual connection ensures
/// output remains bounded.
#[test]
fn test_gqa_repeat_kv_detailed_ibp() {
    let def = build_glm_ocr_gqa_repeat_kv_kernel();
    let bindings = glm_ocr_gqa_repeat_kv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR GQA repeat_kv");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "GQA repeat_kv output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR GQA repeat_kv 4:1 IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Residual preserves bounded output
    assert!(
        lo_min > -100.0,
        "GQA repeat_kv lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 18. Causal mask + RoPE + attention IBP
// ===========================================================================

/// Build causal attention with rotary position embedding.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Q/K projections -> RoPE (element-wise multiply with cos/sin PE)
/// -> causal attention -> output projection -> residual.
fn build_glm_ocr_causal_rope_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_causal_rope_attention");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Q/K/V projections
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);

    // Apply RoPE to Q and K: q_rot = q * cos_pe + q * sin_pe (simplified)
    let cos_pe = b.add_input("cos_pe", &[SEQ_LEN, HIDDEN_DIM]);
    let sin_pe = b.add_input("sin_pe", &[SEQ_LEN, HIDDEN_DIM]);

    let q_cos = b.add_binary_mul(q, cos_pe, &shape);
    let q_sin = b.add_binary_mul(q, sin_pe, &shape);
    let q_rot = b.add_binary_add(q_cos, q_sin, &shape);

    let k_cos = b.add_binary_mul(k, cos_pe, &shape);
    let k_sin = b.add_binary_mul(k, sin_pe, &shape);
    let k_rot = b.add_binary_add(k_cos, k_sin, &shape);

    // Causal attention with rotated Q/K
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q_rot, k_rot, v, AttentionMask::Causal, Some(scale), &shape);

    // Output projection + residual
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(input, projected, &shape);

    b.build(out)
        .expect("valid GLM-OCR causal RoPE attention kernel")
}

/// Bindings for causal RoPE attention.
fn glm_ocr_causal_rope_attention_bindings() -> Vec<TensorParamBinding> {
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    // Generate representative cos/sin PE values
    let n = SEQ_LEN * HIDDEN_DIM;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for t in 0..SEQ_LEN {
        for d in 0..HIDDEN_DIM {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * (d / 2) as f64 / HIDDEN_DIM as f64);
            cos_data.push(freq.cos() as f32);
            sin_data.push(freq.sin() as f32);
        }
    }
    let cos_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), cos_data).expect("valid cos shape");
    let sin_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), sin_data).expect("valid sin shape");

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(cos_pe), // cos_pe (shared for Q and K)
        TensorParamBinding::ConstantTensor(sin_pe), // sin_pe (shared for Q and K)
        TensorParamBinding::ConstantTensor(qkv_w),         // out_weight
    ]
}

/// IBP bounds through causal attention with rotary position embedding.
///
/// RoPE multiplies Q/K by cos/sin values in [-1, 1], which scales
/// but doesn't unboundedly amplify. Combined with causal masking
/// and residual connection, output should remain bounded.
#[test]
fn test_causal_rope_attention_ibp() {
    let def = build_glm_ocr_causal_rope_attention_kernel();
    let bindings = glm_ocr_causal_rope_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR causal RoPE attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "causal RoPE attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR causal RoPE attention IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 19. Multi-token parallel decode IBP
// ===========================================================================

/// Build a parallel multi-token decode kernel.
///
/// Input: `[SEQ_LEN * 2, HIDDEN_DIM]` (Variable, 2x sequence for parallel decode).
/// Output: `[SEQ_LEN * 2, VOCAB_SIZE]` (probability distribution for each token).
///
/// Models the case where N tokens are decoded simultaneously with
/// shared attention context. Uses standard attention (no causal mask
/// since all positions attend to all) + LM head.
fn build_glm_ocr_multi_token_parallel_kernel() -> TensorKernelDef {
    let par_seq = SEQ_LEN * 2; // 8 tokens decoded in parallel
    let mut b = TensorBlockBuilder::new("glm_ocr_multi_token_parallel");

    let input = b.add_input("hidden", &[par_seq, HIDDEN_DIM]);
    let shape = [par_seq, HIDDEN_DIM];
    let ffn_shape = [par_seq, FFN_DIM];

    // Pre-attention RMSNorm
    let n1_eps = b.add_input("norm1_eps", &[1]);
    let n1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, n1_eps, 1, n1_w, &shape);

    // Standard (not causal) attention -- all positions attend to all
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm + SwiGLU
    let n2_eps = b.add_input("norm2_eps", &[1]);
    let n2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    let res2 = b.add_binary_add(res1, ffn_out, &shape);

    // LM head: Linear -> softmax
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(res2, lm_w, None, &[par_seq, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[par_seq, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid GLM-OCR multi-token parallel kernel")
}

/// Bindings for multi-token parallel decode.
fn glm_ocr_multi_token_parallel_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5), // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(qkv_w), // out_weight
        TensorParamBinding::ConstantScalar(1e-5), // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w), // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w), // gate_weight
        TensorParamBinding::ConstantTensor(up_w), // up_weight
        TensorParamBinding::ConstantTensor(down_w), // down_weight
        TensorParamBinding::ConstantTensor(lm_w), // lm_head_weight
    ]
}

/// IBP bounds through multi-token parallel decode.
///
/// Decoding 2x sequence length in parallel. Standard (non-causal)
/// attention means all positions attend to all others. Output
/// via softmax is in [0, 1].
#[test]
fn test_multi_token_parallel_ibp() {
    let par_seq = SEQ_LEN * 2;
    let def = build_glm_ocr_multi_token_parallel_kernel();
    let bindings = glm_ocr_multi_token_parallel_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[par_seq, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR multi-token parallel");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[par_seq, VOCAB_SIZE],
        "multi-token parallel output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR multi-token parallel IBP (hidden [-1,1], seq={par_seq}): bounds=[{lo_min}, {hi_max}]"
    );

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "multi-token parallel lower must be >= 0 (softmax), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "multi-token parallel upper must be <= 1 (softmax), got {hi_max}"
    );
}

// ===========================================================================
// 20. RMSNorm -> GQA -> RMSNorm -> SwiGLU CROWN (full decoder layer)
// ===========================================================================

/// CROWN bounds through full decoder layer subcomponents.
///
/// Verifies that CROWN linearization works through the complete
/// decoder layer composition: RMSNorm -> GQA attention -> residual ->
/// RMSNorm -> SwiGLU FFN -> residual. Reuses the existing decoder
/// layer builder but runs CROWN instead of IBP.
#[test]
fn test_rmsnorm_gqa_rmsnorm_swiglu_crown() {
    let def = build_glm_ocr_decoder_layer_kernel();
    let bindings = glm_ocr_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Use tighter input bounds for CROWN stability through deep composition
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR RMSNorm->GQA->RMSNorm->SwiGLU CROWN (hidden [-0.5,0.5]): \
         method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 21. Token embedding projection IBP
// ===========================================================================

/// Build a token embedding -> projection -> RMSNorm kernel.
///
/// Input: `[SEQ_LEN]` (Variable, token indices).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Tests embedding bounds propagation through normalization.
/// Embedding output is bounded by weight table range; RMSNorm
/// normalizes the magnitude.
fn build_glm_ocr_token_embedding_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_token_embedding_projection");

    let input = b.add_input("token_ids", &[SEQ_LEN]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Embedding lookup
    let emb_w = b.add_input("embedding_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let embedded = b.add_embedding(input, emb_w, &shape);

    // Linear projection
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_bias", &[HIDDEN_DIM]);
    let projected = b.add_linear(embedded, proj_w, Some(proj_b), &shape);

    // RMSNorm
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(projected, eps, 1, norm_w, &shape);

    b.build(out)
        .expect("valid GLM-OCR token embedding projection kernel")
}

/// Bindings for token embedding -> projection -> RMSNorm.
fn glm_ocr_token_embedding_projection_bindings() -> Vec<TensorParamBinding> {
    let emb_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(emb_w),
        TensorParamBinding::ConstantTensor(proj_w),
        TensorParamBinding::ConstantTensor(proj_b),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(norm_w),
    ]
}

/// IBP bounds through token embedding -> projection -> RMSNorm.
///
/// Embedding with bounded weights -> linear projection -> RMSNorm.
/// Tests that RMSNorm constrains the projected embedding output
/// to a bounded range.
#[test]
fn test_token_embedding_projection_ibp() {
    let def = build_glm_ocr_token_embedding_projection_kernel();
    let bindings = glm_ocr_token_embedding_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Token indices bounded in [0, VOCAB_SIZE-1]
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), (VOCAB_SIZE - 1) as f32),
    )
    .expect("valid index bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR token embedding projection");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "token embedding projection output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR token embedding projection IBP (indices [0,{}]): bounds=[{lo_min}, {hi_max}]",
        VOCAB_SIZE - 1
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 22. Full 24-layer decoder -> LM head IBP (realistic depth estimate)
// ===========================================================================

/// Build a 24-layer decoder stack with LM head.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution).
///
/// 24 layers is the standard for GLM-4V-9B. This tests that IBP
/// bounds remain finite (not diverging to infinity) after propagating
/// through a realistic number of decoder layers with small weights.
fn build_glm_ocr_24_layer_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_24_layer_decoder");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer in 0..24 {
        let prefix = format!("l{layer}");

        // Pre-attention RMSNorm
        let n1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let n1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed_a = b.add_rms_norm(current, n1_eps, 1, n1_w, &shape);

        // Self-attention
        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed_a, q_w, None, &shape);
        let k = b.add_linear(normed_a, k_w, None, &shape);
        let v = b.add_linear(normed_a, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res_a = b.add_binary_add(current, attn_out, &shape);

        // Pre-FFN RMSNorm
        let n2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let n2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed_b = b.add_rms_norm(res_a, n2_eps, 1, n2_w, &shape);

        // SwiGLU FFN
        let gate_w = b.add_input(&format!("{prefix}_gate_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let up_w = b.add_input(&format!("{prefix}_up_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("{prefix}_down_weight"), &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed_b, gate_w, None, &ffn_shape);
        let gate_sig = b.add_sigmoid(gate, &ffn_shape);
        let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
        let up = b.add_linear(normed_b, up_w, None, &ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
        let ffn_out = b.add_linear(hidden, down_w, None, &shape);
        current = b.add_binary_add(res_a, ffn_out, &shape);
    }

    // LM Head: RMSNorm -> Linear -> softmax
    let fn_eps = b.add_input("final_norm_eps", &[1]);
    let fn_w = b.add_input("final_norm_weight", &[HIDDEN_DIM]);
    let final_normed = b.add_rms_norm(current, fn_eps, 1, fn_w, &shape);

    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(final_normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid GLM-OCR 24-layer decoder kernel")
}

/// Bindings for 24-layer decoder stack.
fn glm_ocr_24_layer_decoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _ in 0..24 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gate_weight
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // up_weight
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // down_weight
    }

    // LM Head
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // final_norm_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w)); // final_norm_weight
    bindings.push(TensorParamBinding::ConstantTensor(lm_w)); // lm_head_weight

    bindings
}

/// IBP bounds through full 24-layer decoder stack with LM head.
///
/// Realistic depth: GLM-4V-9B has 40 layers. 24 layers is the standard
/// depth for many GLM variants. With WEIGHT_MAG=0.02 and residual
/// connections, bounds should remain finite. This is a depth stress
/// test -- bounds width may be large but must not diverge to infinity.
#[test]
fn test_full_24layer_decoder_lm_head_ibp() {
    let def = build_glm_ocr_24_layer_decoder_kernel();
    let bindings = glm_ocr_24_layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR 24-layer decoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "24-layer decoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR 24-layer decoder IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(
        lo_min.is_finite(),
        "lower bound must be finite after 24 layers"
    );
    assert!(
        hi_max.is_finite(),
        "upper bound must be finite after 24 layers"
    );
    // Final softmax: output in [0, 1]
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "24-layer decoder lower must be >= 0 (softmax output), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "24-layer decoder upper must be <= 1 (softmax output), got {hi_max}"
    );
}

// ===========================================================================
// 23. 8-layer decoder stack IBP (intermediate depth stress test)
// ===========================================================================

/// Build an 8-layer decoder stack without LM head.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// 8 layers sits between the existing 4-layer and 24-layer tests,
/// providing a mid-depth residual accumulation stress test.
fn build_glm_ocr_eight_layer_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_eight_layer_decoder");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer in 0..8 {
        let prefix = format!("l{layer}");

        let n1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let n1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed_a = b.add_rms_norm(current, n1_eps, 1, n1_w, &shape);

        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed_a, q_w, None, &shape);
        let k = b.add_linear(normed_a, k_w, None, &shape);
        let v = b.add_linear(normed_a, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res_a = b.add_binary_add(current, attn_out, &shape);

        let n2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let n2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed_b = b.add_rms_norm(res_a, n2_eps, 1, n2_w, &shape);

        let gate_w = b.add_input(&format!("{prefix}_gate_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let up_w = b.add_input(&format!("{prefix}_up_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("{prefix}_down_weight"), &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed_b, gate_w, None, &ffn_shape);
        let gate_sig = b.add_sigmoid(gate, &ffn_shape);
        let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
        let up = b.add_linear(normed_b, up_w, None, &ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
        let ffn_out = b.add_linear(hidden, down_w, None, &shape);
        current = b.add_binary_add(res_a, ffn_out, &shape);
    }

    b.build(current)
        .expect("valid GLM-OCR eight-layer decoder kernel")
}

/// Bindings for 8-layer decoder stack.
fn glm_ocr_eight_layer_decoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable];

    for _ in 0..8 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone()));
    }

    bindings
}

/// IBP bounds through 8-layer decoder stack.
///
/// Mid-depth stress test: 8 chained decoder layers with residual
/// connections. Verifies bounds remain finite and non-vacuous at
/// intermediate depth between 4-layer and 24-layer tests.
#[test]
fn test_decoder_eight_layer_stack_ibp() {
    let def = build_glm_ocr_eight_layer_decoder_kernel();
    let bindings = glm_ocr_eight_layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR eight-layer decoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "eight-layer decoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR eight-layer decoder IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(
        lo_min.is_finite(),
        "lower bound must be finite after 8 layers"
    );
    assert!(
        hi_max.is_finite(),
        "upper bound must be finite after 8 layers"
    );
}

/// CROWN bounds through 8-layer decoder stack with tighter inputs.
///
/// Verifies that CROWN linearization can propagate through 8 layers.
/// Uses tighter input bounds ([-0.5, 0.5]) for CROWN stability.
#[test]
fn test_decoder_eight_layer_stack_crown() {
    let def = build_glm_ocr_eight_layer_decoder_kernel();
    let bindings = glm_ocr_eight_layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR eight-layer decoder CROWN (hidden [-0.5,0.5]): \
         method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 24. 4-layer decoder + LM head IBP (end-to-end at 4 layers)
// ===========================================================================

/// Build a 4-layer decoder stack with LM head (RMSNorm -> Linear -> softmax).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution).
///
/// Tests end-to-end from hidden states through 4 decoder layers to
/// vocabulary distribution, verifying that softmax output is in [0, 1].
fn build_glm_ocr_four_layer_lm_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_four_layer_lm_head");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer in 0..4 {
        let prefix = format!("l{layer}");

        let n1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let n1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed_a = b.add_rms_norm(current, n1_eps, 1, n1_w, &shape);

        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed_a, q_w, None, &shape);
        let k = b.add_linear(normed_a, k_w, None, &shape);
        let v = b.add_linear(normed_a, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res_a = b.add_binary_add(current, attn_out, &shape);

        let n2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let n2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed_b = b.add_rms_norm(res_a, n2_eps, 1, n2_w, &shape);

        let gate_w = b.add_input(&format!("{prefix}_gate_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let up_w = b.add_input(&format!("{prefix}_up_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("{prefix}_down_weight"), &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed_b, gate_w, None, &ffn_shape);
        let gate_sig = b.add_sigmoid(gate, &ffn_shape);
        let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
        let up = b.add_linear(normed_b, up_w, None, &ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
        let ffn_out = b.add_linear(hidden, down_w, None, &shape);
        current = b.add_binary_add(res_a, ffn_out, &shape);
    }

    // LM Head
    let fn_eps = b.add_input("final_norm_eps", &[1]);
    let fn_w = b.add_input("final_norm_weight", &[HIDDEN_DIM]);
    let final_normed = b.add_rms_norm(current, fn_eps, 1, fn_w, &shape);

    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(final_normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid GLM-OCR four-layer LM head kernel")
}

/// Bindings for 4-layer decoder + LM head.
fn glm_ocr_four_layer_lm_head_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable];

    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone()));
    }

    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(norm_w));
    bindings.push(TensorParamBinding::ConstantTensor(lm_w));

    bindings
}

/// IBP through 4-layer decoder + LM head. Output must be in [0, 1].
#[test]
fn test_four_layer_lm_head_ibp() {
    let def = build_glm_ocr_four_layer_lm_head_kernel();
    let bindings = glm_ocr_four_layer_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR four-layer LM head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "four-layer LM head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR four-layer LM head IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "four-layer LM head lower must be >= 0 (softmax), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "four-layer LM head upper must be <= 1 (softmax), got {hi_max}"
    );
}

// ===========================================================================
// 25. MTP 2-step chain with intermediate RMSNorm (deeper MTP)
// ===========================================================================

/// Build a 2-step MTP chain where each step includes RMSNorm
/// before the projection, modeling the full MTP prediction head
/// architecture from GLM-4V.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (final step distribution).
///
/// Step 1: RMSNorm -> Linear(HIDDEN -> VOCAB) -> softmax
/// Step 2: Linear(VOCAB -> HIDDEN) -> RMSNorm -> Linear(HIDDEN -> VOCAB) -> softmax
fn build_glm_ocr_mtp_normed_chain_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_mtp_normed_chain");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];
    let vocab_shape = [SEQ_LEN, VOCAB_SIZE];

    // Step 1: RMSNorm -> Linear -> softmax
    let n1_eps = b.add_input("s1_norm_eps", &[1]);
    let n1_w = b.add_input("s1_norm_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, n1_eps, 1, n1_w, &hidden_shape);

    let lm_w1 = b.add_input("lm_head_weight_1", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits1 = b.add_linear(normed1, lm_w1, None, &vocab_shape);
    let _probs1 = b.add_softmax(logits1, 1, &vocab_shape);

    // Step 2: project back -> RMSNorm -> Linear -> softmax
    let down_w = b.add_input("down_proj_weight", &[HIDDEN_DIM, VOCAB_SIZE]);
    let hidden2 = b.add_linear(logits1, down_w, None, &hidden_shape);

    let n2_eps = b.add_input("s2_norm_eps", &[1]);
    let n2_w = b.add_input("s2_norm_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(hidden2, n2_eps, 1, n2_w, &hidden_shape);

    let lm_w2 = b.add_input("lm_head_weight_2", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits2 = b.add_linear(normed2, lm_w2, None, &vocab_shape);
    let probs2 = b.add_softmax(logits2, 1, &vocab_shape);

    b.build(probs2)
        .expect("valid GLM-OCR MTP normed chain kernel")
}

/// Bindings for MTP normed chain.
fn glm_ocr_mtp_normed_chain_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, VOCAB_SIZE]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5), // s1_norm_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // s1_norm_weight
        TensorParamBinding::ConstantTensor(lm_w.clone()), // lm_head_weight_1
        TensorParamBinding::ConstantTensor(down_w), // down_proj_weight
        TensorParamBinding::ConstantScalar(1e-5), // s2_norm_eps
        TensorParamBinding::ConstantTensor(norm_w), // s2_norm_weight
        TensorParamBinding::ConstantTensor(lm_w), // lm_head_weight_2
    ]
}

/// IBP through 2-step MTP chain with intermediate RMSNorm.
///
/// Each step includes normalization before projection. Final output
/// through softmax is in [0, 1].
#[test]
fn test_mtp_normed_chain_ibp() {
    let def = build_glm_ocr_mtp_normed_chain_kernel();
    let bindings = glm_ocr_mtp_normed_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR MTP normed chain");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "MTP normed chain output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR MTP normed chain IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "MTP normed chain lower must be >= 0 (softmax), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "MTP normed chain upper must be <= 1 (softmax), got {hi_max}"
    );
}

/// CROWN through 2-step MTP chain with intermediate RMSNorm.
#[test]
fn test_mtp_normed_chain_crown() {
    let def = build_glm_ocr_mtp_normed_chain_kernel();
    let bindings = glm_ocr_mtp_normed_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR MTP normed chain CROWN (hidden [-0.5,0.5]): \
         method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 26. 3-step MTP chain CROWN (complement to existing IBP test)
// ===========================================================================

/// CROWN bounds through 3-step MTP prediction head chain.
///
/// Tests that CROWN linearization works through 3 sequential
/// Linear -> softmax heads, which involve repeated softmax
/// linearization.
#[test]
fn test_mtp_three_head_chain_crown() {
    let def = build_glm_ocr_mtp_three_head_kernel();
    let bindings = glm_ocr_mtp_three_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR MTP three-head CROWN (hidden [-0.5,0.5]): \
         method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "three-head CROWN lower must be >= 0 (softmax), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "three-head CROWN upper must be <= 1 (softmax), got {hi_max}"
    );
}

// ===========================================================================
// 27. Embedding -> RoPE -> attention composition IBP
// ===========================================================================

/// Build embedding -> rotary position encoding -> attention.
///
/// Input: `[SEQ_LEN]` (Variable, token indices).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Full composition: embedding lookup -> Q/K projection -> RoPE ->
/// causal attention -> output projection.
fn build_glm_ocr_embedding_rope_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_embedding_rope_attention");

    let input = b.add_input("token_ids", &[SEQ_LEN]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Embedding lookup
    let emb_w = b.add_input("embedding_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let embedded = b.add_embedding(input, emb_w, &shape);

    // Q/K/V projections
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(embedded, q_w, None, &shape);
    let k = b.add_linear(embedded, k_w, None, &shape);
    let v = b.add_linear(embedded, v_w, None, &shape);

    // RoPE on Q and K
    let cos_pe = b.add_input("cos_pe", &[SEQ_LEN, HIDDEN_DIM]);
    let sin_pe = b.add_input("sin_pe", &[SEQ_LEN, HIDDEN_DIM]);

    let q_cos = b.add_binary_mul(q, cos_pe, &shape);
    let q_sin = b.add_binary_mul(q, sin_pe, &shape);
    let q_rot = b.add_binary_add(q_cos, q_sin, &shape);

    let k_cos = b.add_binary_mul(k, cos_pe, &shape);
    let k_sin = b.add_binary_mul(k, sin_pe, &shape);
    let k_rot = b.add_binary_add(k_cos, k_sin, &shape);

    // Causal attention
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q_rot, k_rot, v, AttentionMask::Causal, Some(scale), &shape);

    // Output projection
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(attn, out_w, None, &shape);

    b.build(out)
        .expect("valid GLM-OCR embedding RoPE attention kernel")
}

/// Bindings for embedding -> RoPE -> attention.
fn glm_ocr_embedding_rope_attention_bindings() -> Vec<TensorParamBinding> {
    let emb_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    let n = SEQ_LEN * HIDDEN_DIM;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for t in 0..SEQ_LEN {
        for d in 0..HIDDEN_DIM {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * (d / 2) as f64 / HIDDEN_DIM as f64);
            cos_data.push(freq.cos() as f32);
            sin_data.push(freq.sin() as f32);
        }
    }
    let cos_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), cos_data).expect("valid cos shape");
    let sin_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), sin_data).expect("valid sin shape");

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(emb_w), // embedding_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(cos_pe), // cos_pe
        TensorParamBinding::ConstantTensor(sin_pe), // sin_pe
        TensorParamBinding::ConstantTensor(qkv_w), // out_weight
    ]
}

/// IBP through embedding -> RoPE -> causal attention.
///
/// Verifies the full composition: token embedding lookup produces
/// bounded vectors, RoPE rotates Q/K with cos/sin in [-1, 1],
/// causal attention produces bounded output.
#[test]
fn test_embedding_rope_attention_ibp() {
    let def = build_glm_ocr_embedding_rope_attention_kernel();
    let bindings = glm_ocr_embedding_rope_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR embedding RoPE attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "embedding RoPE attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR embedding->RoPE->attention IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 28. GQA at scale: 16 heads, 2 KV heads (realistic head count)
// ===========================================================================

/// Larger GQA dimensions: 16 query heads, 2 KV heads.
///
/// Constants for this test:
/// - HIDDEN_DIM_LARGE = 128 (16 heads * 8 head_dim)
/// - KV_DIM_LARGE = 16 (2 KV heads * 8 head_dim)
/// - HEAD_DIM_LARGE = 8
///
/// This models a more realistic GQA ratio (8:1) than the base
/// test's 2:1.
const HIDDEN_DIM_LARGE: usize = 128;
const NUM_HEADS_LARGE: usize = 16;
const NUM_KV_HEADS_LARGE: usize = 2;
const HEAD_DIM_LARGE: usize = HIDDEN_DIM_LARGE / NUM_HEADS_LARGE; // 8
const KV_DIM_LARGE: usize = NUM_KV_HEADS_LARGE * HEAD_DIM_LARGE; // 16

fn build_glm_ocr_gqa_16h_2kv_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_gqa_16h_2kv");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM_LARGE]);
    let shape = [SEQ_LEN, HIDDEN_DIM_LARGE];

    // Q: full hidden dim (16 heads)
    let q_w = b.add_input("q_proj_weight", &[HIDDEN_DIM_LARGE, HIDDEN_DIM_LARGE]);
    // K/V: 2 KV heads only
    let k_w = b.add_input("k_proj_weight", &[KV_DIM_LARGE, HIDDEN_DIM_LARGE]);
    let v_w = b.add_input("v_proj_weight", &[KV_DIM_LARGE, HIDDEN_DIM_LARGE]);

    let _q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, KV_DIM_LARGE]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, KV_DIM_LARGE]);

    // Project Q down to KV dim for verification tractability
    let q_down_w = b.add_input("q_down_weight", &[KV_DIM_LARGE, HIDDEN_DIM_LARGE]);
    let q_down = b.add_linear(input, q_down_w, None, &[SEQ_LEN, KV_DIM_LARGE]);

    let scale = 1.0 / (HEAD_DIM_LARGE as f32).sqrt();
    let attn_out = b.add_attention(
        q_down,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM_LARGE],
    );

    // Project back to full hidden dim
    let out_w = b.add_input("out_proj_weight", &[HIDDEN_DIM_LARGE, KV_DIM_LARGE]);
    let projected = b.add_linear(attn_out, out_w, None, &shape);

    // Residual
    let out = b.add_binary_add(input, projected, &shape);

    b.build(out).expect("valid GLM-OCR GQA 16h/2kv kernel")
}

fn glm_ocr_gqa_16h_2kv_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM_LARGE, HIDDEN_DIM_LARGE]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM_LARGE, HIDDEN_DIM_LARGE]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM_LARGE, HIDDEN_DIM_LARGE]), WEIGHT_MAG);
    let q_down_w = ArrayD::from_elem(IxDyn(&[KV_DIM_LARGE, HIDDEN_DIM_LARGE]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM_LARGE, KV_DIM_LARGE]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(q_w),
        TensorParamBinding::ConstantTensor(k_w),
        TensorParamBinding::ConstantTensor(v_w),
        TensorParamBinding::ConstantTensor(q_down_w),
        TensorParamBinding::ConstantTensor(out_w),
    ]
}

/// IBP through GQA with 16 query heads and 2 KV heads (8:1 ratio).
///
/// Realistic head configuration. The high GQA ratio means K/V
/// projections are much smaller than Q, producing asymmetric
/// intermediate bounds. Residual connection stabilizes output.
#[test]
fn test_gqa_16h_2kv_ibp() {
    let def = build_glm_ocr_gqa_16h_2kv_kernel();
    let bindings = glm_ocr_gqa_16h_2kv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM_LARGE], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR GQA 16h/2kv");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM_LARGE],
        "GQA 16h/2kv output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR GQA 16h/2kv IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through GQA with 16 query heads and 2 KV heads.
#[test]
fn test_gqa_16h_2kv_crown() {
    let def = build_glm_ocr_gqa_16h_2kv_kernel();
    let bindings = glm_ocr_gqa_16h_2kv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM_LARGE], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM_LARGE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR GQA 16h/2kv CROWN (hidden [-0.5,0.5]): \
         method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 29. Residual accumulation depth test: 4 layers with narrowing input
// ===========================================================================

/// Verify bounds width growth rate through 4 decoder layers.
///
/// Tests the same 4-layer decoder with three different input bound
/// widths to verify that residual accumulation does not cause
/// super-exponential blow-up: tighter inputs -> tighter outputs.
#[test]
fn test_residual_accumulation_monotonic_tightening() {
    let def = build_glm_ocr_four_layer_decoder_kernel();
    let bindings = glm_ocr_four_layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Run with three different input ranges
    let input_wide = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let input_mid = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);
    let input_tight = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);

    let out_wide = graph.propagate_ibp(&input_wide).expect("IBP wide");
    let out_mid = graph.propagate_ibp(&input_mid).expect("IBP mid");
    let out_tight = graph.propagate_ibp(&input_tight).expect("IBP tight");

    let (lo_w, hi_w) = bounds_min_max(&out_wide);
    let (lo_m, hi_m) = bounds_min_max(&out_mid);
    let (lo_t, hi_t) = bounds_min_max(&out_tight);

    let width_wide = hi_w - lo_w;
    let width_mid = hi_m - lo_m;
    let width_tight = hi_t - lo_t;

    eprintln!(
        "Residual accumulation: wide={width_wide:.4}, mid={width_mid:.4}, tight={width_tight:.4}"
    );

    // Monotonicity: tighter inputs should produce tighter (or equal) outputs
    assert!(
        width_tight <= width_mid + 1e-6,
        "tight input ({width_tight:.6}) should produce narrower bounds than mid ({width_mid:.6})"
    );
    assert!(
        width_mid <= width_wide + 1e-6,
        "mid input ({width_mid:.6}) should produce narrower bounds than wide ({width_wide:.6})"
    );

    // All must be finite
    assert!(width_wide.is_finite(), "wide bounds width must be finite");
    assert!(width_mid.is_finite(), "mid bounds width must be finite");
    assert!(width_tight.is_finite(), "tight bounds width must be finite");
}

// ===========================================================================
// 30. Full decoder stack verify_and_record (2-layer, CROWN)
// ===========================================================================

/// CROWN through full 2-layer decoder stack with LM head.
///
/// Complement to test_full_decoder_stack_ibp. Verifies CROWN
/// linearization can handle the full decoder -> LM head -> softmax
/// pipeline.
#[test]
fn test_full_decoder_stack_crown() {
    let def = build_glm_ocr_full_decoder_stack_kernel();
    let bindings = glm_ocr_full_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR full decoder stack CROWN (hidden [-0.5,0.5]): \
         method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "full decoder stack CROWN lower must be >= 0 (softmax), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "full decoder stack CROWN upper must be <= 1 (softmax), got {hi_max}"
    );
}

// ===========================================================================
// 31. 4-layer decoder CROWN (complement to IBP-only test)
// ===========================================================================

/// CROWN bounds through 4-layer decoder (no LM head).
///
/// Verifies CROWN linearization stability through 4 chained decoder
/// layers with residual connections. Uses tighter inputs for stability.
#[test]
fn test_decoder_four_layer_stack_crown() {
    let def = build_glm_ocr_four_layer_decoder_kernel();
    let bindings = glm_ocr_four_layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR four-layer decoder CROWN (hidden [-0.5,0.5]): \
         method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 32. MTP 2-step chain verify_and_record
// ===========================================================================

/// Verify and record 2-step MTP chain.
#[test]
fn test_mtp_multi_step_verify_and_record() {
    let def = build_glm_ocr_mtp_multi_step_kernel();
    let bindings = glm_ocr_mtp_multi_step_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "glm_ocr_mtp_multi_step");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ===========================================================================
// 33. Causal RoPE attention CROWN
// ===========================================================================

/// CROWN bounds through causal attention with rotary position embedding.
///
/// Complement to test_causal_rope_attention_ibp. RoPE element-wise
/// multiplications require CROWN linearization for tighter bounds.
#[test]
fn test_causal_rope_attention_crown() {
    let def = build_glm_ocr_causal_rope_attention_kernel();
    let bindings = glm_ocr_causal_rope_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR causal RoPE attention CROWN (hidden [-0.5,0.5]): \
         method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 34. Decoder layer + MTP head end-to-end IBP
// ===========================================================================

/// Build a single decoder layer followed by MTP head.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution).
///
/// Tests the boundary between decoder hidden states and the
/// prediction head: decoder layer -> RMSNorm -> Linear -> softmax.
fn build_glm_ocr_decoder_mtp_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_decoder_mtp");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Decoder layer
    let n1_eps = b.add_input("norm1_eps", &[1]);
    let n1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, n1_eps, 1, n1_w, &shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

    let n2_eps = b.add_input("norm2_eps", &[1]);
    let n2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    let res2 = b.add_binary_add(res1, ffn_out, &shape);

    // MTP head: RMSNorm -> Linear -> softmax
    let fn_eps = b.add_input("final_norm_eps", &[1]);
    let fn_w = b.add_input("final_norm_weight", &[HIDDEN_DIM]);
    let final_normed = b.add_rms_norm(res2, fn_eps, 1, fn_w, &shape);

    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(final_normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid GLM-OCR decoder + MTP kernel")
}

/// Bindings for decoder layer + MTP head.
fn glm_ocr_decoder_mtp_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5), // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(qkv_w), // out_weight
        TensorParamBinding::ConstantScalar(1e-5), // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w), // gate_weight
        TensorParamBinding::ConstantTensor(up_w), // up_weight
        TensorParamBinding::ConstantTensor(down_w), // down_weight
        TensorParamBinding::ConstantScalar(1e-5), // final_norm_eps
        TensorParamBinding::ConstantTensor(norm_w), // final_norm_weight
        TensorParamBinding::ConstantTensor(lm_w), // lm_head_weight
    ]
}

/// IBP through decoder layer + MTP head. Output in [0, 1].
#[test]
fn test_decoder_mtp_ibp() {
    let def = build_glm_ocr_decoder_mtp_kernel();
    let bindings = glm_ocr_decoder_mtp_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR decoder + MTP");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "decoder + MTP output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR decoder+MTP IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "decoder+MTP lower must be >= 0 (softmax), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "decoder+MTP upper must be <= 1 (softmax), got {hi_max}"
    );
}

/// CROWN through decoder layer + MTP head.
#[test]
fn test_decoder_mtp_crown() {
    let def = build_glm_ocr_decoder_mtp_kernel();
    let bindings = glm_ocr_decoder_mtp_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR decoder+MTP CROWN (hidden [-0.5,0.5]): \
         method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "decoder+MTP CROWN lower must be >= 0 (softmax), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "decoder+MTP CROWN upper must be <= 1 (softmax), got {hi_max}"
    );
}

// ===========================================================================
// 35. 24-layer decoder verify_and_record
// ===========================================================================

/// Verify and record 24-layer decoder.
#[test]
fn test_full_24layer_decoder_verify_and_record() {
    let def = build_glm_ocr_24_layer_decoder_kernel();
    let bindings = glm_ocr_24_layer_decoder_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "glm_ocr_24_layer_decoder");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ===========================================================================
// 36. Embedding -> RoPE -> GQA -> SwiGLU (full first-layer composition)
// ===========================================================================

/// Build embedding -> RoPE -> GQA attention -> RMSNorm -> SwiGLU FFN.
///
/// Input: `[SEQ_LEN]` (Variable, token indices).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Complete first-layer composition from token input through all
/// sub-blocks of the first decoder layer.
fn build_glm_ocr_first_layer_full_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_first_layer_full");

    let input = b.add_input("token_ids", &[SEQ_LEN]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Embedding
    let emb_w = b.add_input("embedding_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let embedded = b.add_embedding(input, emb_w, &shape);

    // Pre-attention RMSNorm
    let n1_eps = b.add_input("norm1_eps", &[1]);
    let n1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(embedded, n1_eps, 1, n1_w, &shape);

    // Q/K/V projections
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);

    // Apply RoPE to Q and K
    let cos_pe = b.add_input("cos_pe", &[SEQ_LEN, HIDDEN_DIM]);
    let sin_pe = b.add_input("sin_pe", &[SEQ_LEN, HIDDEN_DIM]);

    let q_cos = b.add_binary_mul(q, cos_pe, &shape);
    let q_sin = b.add_binary_mul(q, sin_pe, &shape);
    let q_rot = b.add_binary_add(q_cos, q_sin, &shape);

    let k_cos = b.add_binary_mul(k, cos_pe, &shape);
    let k_sin = b.add_binary_mul(k, sin_pe, &shape);
    let k_rot = b.add_binary_add(k_cos, k_sin, &shape);

    // Causal attention
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q_rot, k_rot, v, AttentionMask::Causal, Some(scale), &shape);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(embedded, attn_out, &shape);

    // Pre-FFN RMSNorm + SwiGLU
    let n2_eps = b.add_input("norm2_eps", &[1]);
    let n2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    let out = b.add_binary_add(res1, ffn_out, &shape);

    b.build(out).expect("valid GLM-OCR first-layer full kernel")
}

/// Bindings for first-layer full composition.
fn glm_ocr_first_layer_full_bindings() -> Vec<TensorParamBinding> {
    let emb_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let n = SEQ_LEN * HIDDEN_DIM;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for t in 0..SEQ_LEN {
        for d in 0..HIDDEN_DIM {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * (d / 2) as f64 / HIDDEN_DIM as f64);
            cos_data.push(freq.cos() as f32);
            sin_data.push(freq.sin() as f32);
        }
    }
    let cos_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), cos_data).expect("valid cos shape");
    let sin_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), sin_data).expect("valid sin shape");

    vec![
        TensorParamBinding::Variable,                       // token_ids
        TensorParamBinding::ConstantTensor(emb_w),          // embedding_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()),  // q_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()),  // k_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()),  // v_weight
        TensorParamBinding::ConstantTensor(cos_pe),         // cos_pe
        TensorParamBinding::ConstantTensor(sin_pe),         // sin_pe
        TensorParamBinding::ConstantTensor(qkv_w),          // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
    ]
}

/// IBP through embedding -> RoPE -> GQA attention -> RMSNorm -> SwiGLU.
///
/// Full first-layer composition from token input through the complete
/// first decoder layer.
#[test]
fn test_first_layer_full_ibp() {
    let def = build_glm_ocr_first_layer_full_kernel();
    let bindings = glm_ocr_first_layer_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR first-layer full");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "first-layer full output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR first-layer full IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 37. Multi-token parallel decode CROWN
// ===========================================================================

/// CROWN through multi-token parallel decode.
///
/// Complement to test_multi_token_parallel_ibp. Tests CROWN
/// linearization with standard (non-causal) attention over 2x
/// sequence length.
#[test]
fn test_multi_token_parallel_crown() {
    let par_seq = SEQ_LEN * 2;
    let def = build_glm_ocr_multi_token_parallel_kernel();
    let bindings = glm_ocr_multi_token_parallel_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[par_seq, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[par_seq, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR multi-token parallel CROWN (hidden [-0.5,0.5], seq={par_seq}): \
         method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "multi-token parallel CROWN lower must be >= 0 (softmax), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "multi-token parallel CROWN upper must be <= 1 (softmax), got {hi_max}"
    );
}

// ===========================================================================
// 38. Linear projection with GELU activation IBP
// ===========================================================================

/// Build a linear projection followed by GELU activation.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Architecture: Linear(HIDDEN -> FFN) -> GELU -> Linear(FFN -> HIDDEN).
/// Tests GELU activation bounds through a two-layer projection, common in
/// GLM-OCR vision-to-text adapter layers.
fn build_glm_ocr_linear_gelu_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_linear_gelu_projection");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // Up projection + GELU
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_b = b.add_input("up_bias", &[FFN_DIM]);
    let up = b.add_linear(input, up_w, Some(up_b), &ffn_shape);
    let activated = b.add_gelu(up, &ffn_shape);

    // Down projection
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);
    let down_b = b.add_input("down_bias", &[HIDDEN_DIM]);
    let out = b.add_linear(activated, down_w, Some(down_b), &out_shape);

    b.build(out)
        .expect("valid GLM-OCR linear GELU projection kernel")
}

/// Bindings for linear GELU projection.
fn glm_ocr_linear_gelu_projection_bindings() -> Vec<TensorParamBinding> {
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_b = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let down_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantTensor(up_w),   // up_weight
        TensorParamBinding::ConstantTensor(up_b),   // up_bias
        TensorParamBinding::ConstantTensor(down_w), // down_weight
        TensorParamBinding::ConstantTensor(down_b), // down_bias
    ]
}

/// IBP through linear + GELU projection.
///
/// Verifies that GELU activation produces bounded outputs when the
/// input is bounded. GELU is smooth and monotonic for positive inputs,
/// providing tighter bounds than ReLU.
#[test]
fn test_glm_ocr_linear_gelu_projection_ibp() {
    let def = build_glm_ocr_linear_gelu_projection_kernel();
    let bindings = glm_ocr_linear_gelu_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through linear GELU projection");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "linear GELU projection output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR linear GELU projection IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through linear + GELU projection.
///
/// Tests CROWN linearization through GELU (piecewise smooth, good for
/// linear relaxation).
#[test]
fn test_glm_ocr_linear_gelu_projection_crown() {
    let def = build_glm_ocr_linear_gelu_projection_kernel();
    let bindings = glm_ocr_linear_gelu_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR linear GELU projection CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 39. Adapter layers: bottleneck projection IBP
// ===========================================================================

/// Build a bottleneck adapter layer (down-project -> GELU -> up-project).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Architecture: Linear(HIDDEN -> BOTTLENECK) -> GELU ->
/// Linear(BOTTLENECK -> HIDDEN). Models a parameter-efficient adapter
/// layer commonly used to bridge vision and language towers in VLMs.
/// The bottleneck dimension (HIDDEN_DIM/4=16) compresses then expands
/// the representation.
fn build_glm_ocr_bottleneck_adapter_kernel() -> TensorKernelDef {
    let bottleneck_dim = HIDDEN_DIM / 4; // 16
    let mut b = TensorBlockBuilder::new("glm_ocr_bottleneck_adapter");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let bottleneck_shape = [SEQ_LEN, bottleneck_dim];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // Down-project to bottleneck
    let down_w = b.add_input("down_weight", &[bottleneck_dim, HIDDEN_DIM]);
    let down_b = b.add_input("down_bias", &[bottleneck_dim]);
    let down = b.add_linear(input, down_w, Some(down_b), &bottleneck_shape);

    // GELU activation
    let activated = b.add_gelu(down, &bottleneck_shape);

    // Up-project back to hidden dim
    let up_w = b.add_input("up_weight", &[HIDDEN_DIM, bottleneck_dim]);
    let up_b = b.add_input("up_bias", &[HIDDEN_DIM]);
    let out = b.add_linear(activated, up_w, Some(up_b), &out_shape);

    b.build(out)
        .expect("valid GLM-OCR bottleneck adapter kernel")
}

/// Bindings for bottleneck adapter.
fn glm_ocr_bottleneck_adapter_bindings() -> Vec<TensorParamBinding> {
    let bottleneck_dim = HIDDEN_DIM / 4;
    let down_w = ArrayD::from_elem(IxDyn(&[bottleneck_dim, HIDDEN_DIM]), WEIGHT_MAG);
    let down_b = ArrayD::from_elem(IxDyn(&[bottleneck_dim]), 0.0f32);
    let up_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, bottleneck_dim]), WEIGHT_MAG);
    let up_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantTensor(down_w), // down_weight
        TensorParamBinding::ConstantTensor(down_b), // down_bias
        TensorParamBinding::ConstantTensor(up_w),   // up_weight
        TensorParamBinding::ConstantTensor(up_b),   // up_bias
    ]
}

/// IBP through bottleneck adapter.
///
/// The bottleneck compresses the representation to 1/4 dimension then
/// expands it back. Verifies that the information bottleneck naturally
/// constrains output bounds.
#[test]
fn test_glm_ocr_bottleneck_adapter_ibp() {
    let def = build_glm_ocr_bottleneck_adapter_kernel();
    let bindings = glm_ocr_bottleneck_adapter_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through bottleneck adapter");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "bottleneck adapter output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR bottleneck adapter IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through bottleneck adapter.
///
/// Tests CROWN linearization through the compression/expansion bottleneck.
#[test]
fn test_glm_ocr_bottleneck_adapter_crown() {
    let def = build_glm_ocr_bottleneck_adapter_kernel();
    let bindings = glm_ocr_bottleneck_adapter_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR bottleneck adapter CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 40. Projection with residual connection IBP
// ===========================================================================

/// Build a projection layer with a residual connection.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Architecture: Linear(HIDDEN -> FFN) -> GELU -> Linear(FFN -> HIDDEN) + input.
/// The residual connection ensures the output includes the original signal,
/// which typically constrains bounds better than pure projection.
fn build_glm_ocr_projection_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_projection_residual");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // Up projection + GELU
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let activated = b.add_gelu(up, &ffn_shape);

    // Down projection
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);
    let proj = b.add_linear(activated, down_w, None, &out_shape);

    // Residual connection
    let out = b.add_binary_add(input, proj, &out_shape);

    b.build(out)
        .expect("valid GLM-OCR projection with residual kernel")
}

/// Bindings for projection with residual connection.
fn glm_ocr_projection_residual_bindings() -> Vec<TensorParamBinding> {
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantTensor(up_w),   // up_weight
        TensorParamBinding::ConstantTensor(down_w), // down_weight
    ]
}

/// IBP through projection with residual connection.
///
/// The residual skip connection adds the original input to the projection
/// output. This typically produces tighter bounds than projection alone
/// because the residual dominates when the projection produces small values.
#[test]
fn test_glm_ocr_projection_residual_ibp() {
    let def = build_glm_ocr_projection_residual_kernel();
    let bindings = glm_ocr_projection_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through projection with residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "projection residual output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR projection residual IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through projection with residual connection.
///
/// Residual connections improve CROWN linearization because the identity
/// path contributes exact gradients.
#[test]
fn test_glm_ocr_projection_residual_crown() {
    let def = build_glm_ocr_projection_residual_kernel();
    let bindings = glm_ocr_projection_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR projection residual CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 41. RMSNorm + bottleneck adapter + residual IBP
// ===========================================================================

/// Build RMSNorm -> bottleneck adapter -> residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Architecture: RMSNorm -> Linear(HIDDEN -> BOTTLENECK) -> GELU ->
/// Linear(BOTTLENECK -> HIDDEN) -> residual with original input.
/// Combines normalization with bottleneck adapter and residual,
/// modeling the full adapter insertion pattern in GLM-OCR.
fn build_glm_ocr_norm_adapter_residual_kernel() -> TensorKernelDef {
    let bottleneck_dim = HIDDEN_DIM / 4; // 16
    let mut b = TensorBlockBuilder::new("glm_ocr_norm_adapter_residual");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let bottleneck_shape = [SEQ_LEN, bottleneck_dim];

    // RMSNorm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm_eps, 1, norm_w, &shape);

    // Bottleneck: down -> GELU -> up
    let down_w = b.add_input("down_weight", &[bottleneck_dim, HIDDEN_DIM]);
    let down = b.add_linear(normed, down_w, None, &bottleneck_shape);
    let activated = b.add_gelu(down, &bottleneck_shape);

    let up_w = b.add_input("up_weight", &[HIDDEN_DIM, bottleneck_dim]);
    let adapted = b.add_linear(activated, up_w, None, &shape);

    // Residual
    let out = b.add_binary_add(input, adapted, &shape);

    b.build(out)
        .expect("valid GLM-OCR norm adapter residual kernel")
}

/// Bindings for RMSNorm + bottleneck adapter + residual.
fn glm_ocr_norm_adapter_residual_bindings() -> Vec<TensorParamBinding> {
    let bottleneck_dim = HIDDEN_DIM / 4;
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let down_w = ArrayD::from_elem(IxDyn(&[bottleneck_dim, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, bottleneck_dim]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantScalar(1e-5),   // norm_eps
        TensorParamBinding::ConstantTensor(norm_w), // norm_weight
        TensorParamBinding::ConstantTensor(down_w), // down_weight
        TensorParamBinding::ConstantTensor(up_w),   // up_weight
    ]
}

/// IBP through RMSNorm + bottleneck adapter + residual.
///
/// Full adapter pattern: normalize, compress through bottleneck, expand,
/// and add residual. Tests the entire adapter insertion pipeline bounds.
#[test]
fn test_glm_ocr_norm_adapter_residual_ibp() {
    let def = build_glm_ocr_norm_adapter_residual_kernel();
    let bindings = glm_ocr_norm_adapter_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through norm adapter residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "norm adapter residual output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR norm adapter residual IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through RMSNorm + bottleneck adapter + residual.
///
/// Tests CROWN through RMSNorm -> GELU bottleneck -> residual composition.
#[test]
fn test_glm_ocr_norm_adapter_residual_crown() {
    let def = build_glm_ocr_norm_adapter_residual_kernel();
    let bindings = glm_ocr_norm_adapter_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR norm adapter residual CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}
