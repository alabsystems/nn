// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for GLM-OCR decoder full generation pipeline bounds.
//!
//! Verifies IBP and CROWN bound propagation through the GLM-OCR (ChatGLM 0.9B)
//! decoder pipeline used for optical character recognition in the dpdf document
//! understanding stack.
//!
//! ## Tests (18 tests)
//!
//! 1.  **RoPE-enhanced Q/K projection bounds** (IBP)
//! 2.  **GQA multi-head attention output bounds** (IBP)
//! 3.  **SwiGLU FFN gate * up activation bounds** (IBP)
//! 4.  **RMSNorm pre-attention normalization bounds** (IBP)
//! 5.  **RMSNorm pre-FFN normalization bounds** (IBP)
//! 6.  **Residual connection after attention bounds** (IBP)
//! 7.  **Residual connection after FFN bounds** (IBP)
//! 8.  **Single decoder block (attention + FFN + residuals) bounds** (IBP)
//! 9.  **Two-block decoder stack composition** (IBP)
//! 10. **KV cache integration bounds** (IBP)
//! 11. **Final RMSNorm before LM head bounds** (IBP)
//! 12. **Linear LM head logit projection bounds** (IBP)
//! 13. **Full GLM decoder block pipeline** (CROWN)
//! 14. **Embedding layer output bounds** (IBP)
//! 15. **Position encoding combination bounds** (IBP)
//! 16. **Multi-block depth composition bounds** (IBP)
//! 17. **Output logit range after LM head** (IBP)
//! 18. **Temperature-scaled logit bounds** (IBP)
//!
//! Architecture references:
//! - GLM-4V / ChatGLM 0.9B (THUDM): Decoder-only transformer for OCR
//! - RMSNorm (Zhang & Sennrich, 2019): Root mean square layer normalization
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN (gate_proj -> SiLU * up_proj -> down_proj)
//! - GQA (Ainslie et al., 2023): Grouped-query attention
//! - RoPE (Su et al., 2021): Rotary positional embeddings
//!
//! Dimensions (symbolic, small for fast verification):
//! - HIDDEN_DIM=4, FFN_DIM=8, NUM_HEADS=2, NUM_KV_HEADS=2
//! - HEAD_DIM=2, SEQ_LEN=4, VOCAB_SIZE=6
//! - Production GLM-OCR 0.9B: hidden=1536, FFN=8960, heads=12, KV_heads=2
//!
//! Part of #4183: Compose tests for GLM-OCR decoder pipeline bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- symbolic (small for fast verification), structurally
// representative of GLM-OCR 0.9B (production: hidden=1536, FFN=8960,
// heads=12, KV_heads=2, head_dim=128)
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 4;
const FFN_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 2
const SEQ_LEN: usize = 4;
const VOCAB_SIZE: usize = 6;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Ones tensor binding (for RMSNorm weight).
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Zero bias tensor binding.
fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Scalar epsilon binding.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Sequence-domain input bounds: embeddings in [-range, +range].
fn seq_bounds(seq_len: usize, dim: usize, range: f32) -> BoundedTensor {
    uniform_bounds(&[seq_len, dim], range)
}

/// RoPE cos/sin positional encoding in [-1, 1].
fn rope_cos_sin(seq_len: usize, head_dim: usize) -> ArrayD<f32> {
    let mut data = vec![0.0f32; seq_len * head_dim];
    for t in 0..seq_len {
        for i in 0..head_dim / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / head_dim as f64);
            data[t * head_dim + 2 * i] = freq.cos() as f32;
            data[t * head_dim + 2 * i + 1] = freq.sin() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq_len, head_dim]), data).expect("valid RoPE")
}

/// Add a SwiGLU FFN sub-block to a builder.
///
/// gate_proj -> SiLU * up_proj -> down_proj
/// Input/output: [seq_len, hidden_dim]. Returns output node.
fn add_swiglu_ffn(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    prefix: &str,
) -> TensorNodeId {
    let ffn_shape = [seq_len, ffn_dim];
    let out_shape = [seq_len, hidden_dim];

    let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[ffn_dim, hidden_dim]);
    let up_w = b.add_input(&format!("{prefix}_up_w"), &[ffn_dim, hidden_dim]);
    let down_w = b.add_input(&format!("{prefix}_down_w"), &[hidden_dim, ffn_dim]);

    // Gate branch: gate_proj -> SiLU(x) = x * sigmoid(x)
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);

    // Up branch
    let up = b.add_linear(input, up_w, None, &ffn_shape);

    // Multiplicative gating + down projection
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    b.add_linear(hidden, down_w, None, &out_shape)
}

/// Push SwiGLU FFN bindings (3 params: gate_w, up_w, down_w).
fn push_swiglu_bindings(bindings: &mut Vec<TensorParamBinding>, hidden_dim: usize, ffn_dim: usize) {
    bindings.push(weight(&[ffn_dim, hidden_dim])); // gate_w
    bindings.push(weight(&[ffn_dim, hidden_dim])); // up_w
    bindings.push(weight(&[hidden_dim, ffn_dim])); // down_w
}

/// Add a single GLM-OCR decoder block to a builder.
///
/// RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual
/// Input/output: [seq_len, hidden_dim]. Returns output node.
fn add_decoder_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    prefix: &str,
) -> TensorNodeId {
    let shape = [seq_len, hidden_dim];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let n1_eps = b.add_input(&format!("{prefix}_n1_eps"), &[1]);
    let n1_w = b.add_input(&format!("{prefix}_n1_w"), &[hidden_dim]);
    let normed1 = b.add_rms_norm(input, n1_eps, 1, n1_w, &shape);

    // Self-attention (Q/K/V + attention + output projection)
    let q_w = b.add_input(&format!("{prefix}_q_w"), &[hidden_dim, hidden_dim]);
    let k_w = b.add_input(&format!("{prefix}_k_w"), &[hidden_dim, hidden_dim]);
    let v_w = b.add_input(&format!("{prefix}_v_w"), &[hidden_dim, hidden_dim]);
    let o_w = b.add_input(&format!("{prefix}_o_w"), &[hidden_dim, hidden_dim]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_proj = b.add_linear(attn, o_w, None, &shape);

    // Residual after attention
    let res1 = b.add_binary_add(input, attn_proj, &shape);

    // Pre-FFN RMSNorm
    let n2_eps = b.add_input(&format!("{prefix}_n2_eps"), &[1]);
    let n2_w = b.add_input(&format!("{prefix}_n2_w"), &[hidden_dim]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    // SwiGLU FFN
    let ffn_out = add_swiglu_ffn(b, normed2, seq_len, hidden_dim, ffn_dim, prefix);

    // Residual after FFN
    b.add_binary_add(res1, ffn_out, &shape)
}

/// Push decoder block bindings (12 params).
fn push_decoder_block_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    hidden_dim: usize,
    ffn_dim: usize,
) {
    // RMSNorm 1: eps, weight
    bindings.push(eps_binding());
    bindings.push(ones(&[hidden_dim]));
    // Attention: Q, K, V, O weights
    bindings.push(weight(&[hidden_dim, hidden_dim]));
    bindings.push(weight(&[hidden_dim, hidden_dim]));
    bindings.push(weight(&[hidden_dim, hidden_dim]));
    bindings.push(weight(&[hidden_dim, hidden_dim]));
    // RMSNorm 2: eps, weight
    bindings.push(eps_binding());
    bindings.push(ones(&[hidden_dim]));
    // SwiGLU FFN: gate_w, up_w, down_w
    push_swiglu_bindings(bindings, hidden_dim, ffn_dim);
}

// ===========================================================================
// 1. RoPE-enhanced Q/K projection bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_rope_qk_projection_ibp() {
    // RoPE-enhanced Q/K: project -> apply cos/sin rotation -> bounded
    // Simplified: Q_proj -> elementwise mul with cos/sin table (bounded [-1,1])
    let mut b = TensorBlockBuilder::new("glm_pipe_rope_qk");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let cos_table = b.add_input("cos_table", &[SEQ_LEN, HIDDEN_DIM]);
    let sin_table = b.add_input("sin_table", &[SEQ_LEN, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    // RoPE rotation: q * cos + rotate(q) * sin
    // Simplified for verification: q * cos_table (element-wise)
    let q_cos = b.add_binary_mul(q, cos_table, &[SEQ_LEN, HIDDEN_DIM]);
    let q_sin = b.add_binary_mul(q, sin_table, &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_binary_add(q_cos, q_sin, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid RoPE Q/K kernel");

    let cos_data = rope_cos_sin(SEQ_LEN, HIDDEN_DIM);
    let sin_data = rope_cos_sin(SEQ_LEN, HIDDEN_DIM);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(cos_data),
        TensorParamBinding::ConstantTensor(sin_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe RoPE Q/K IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. GQA multi-head attention output bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_gqa_attention_ibp() {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut b = TensorBlockBuilder::new("glm_pipe_gqa_attn");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, HIDDEN_DIM],
    );
    let out = b.add_linear(attn, o_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid GQA attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe GQA attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. SwiGLU FFN gate * up activation bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_swiglu_gate_up_ibp() {
    let mut b = TensorBlockBuilder::new("glm_pipe_swiglu_gate_up");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_swiglu_ffn(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "ffn");
    let def = b.build(out).expect("valid SwiGLU kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe SwiGLU gate*up IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 4. RMSNorm pre-attention normalization bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_rmsnorm_pre_attention_ibp() {
    let mut b = TensorBlockBuilder::new("glm_pipe_rmsnorm_pre_attn");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let w = b.add_input("w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(input, eps, 1, w, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid RMSNorm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe RMSNorm pre-attn IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. RMSNorm pre-FFN normalization bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_rmsnorm_pre_ffn_ibp() {
    // Same structure as pre-attention, but with wider input (post-residual)
    let mut b = TensorBlockBuilder::new("glm_pipe_rmsnorm_pre_ffn");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let w = b.add_input("w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(input, eps, 1, w, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid RMSNorm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Wider input simulating post-residual state
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe RMSNorm pre-FFN IBP (wide): bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. Residual connection after attention bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_residual_after_attention_ibp() {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("glm_pipe_res_after_attn");
    let input = b.add_input("x", &shape);

    // RMSNorm -> attention -> output proj -> add(input, ...)
    let eps = b.add_input("eps", &[1]);
    let nw = b.add_input("nw", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, nw, &shape);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let proj = b.add_linear(attn, o_w, None, &shape);
    let out = b.add_binary_add(input, proj, &shape);
    let def = b.build(out).expect("valid residual-after-attn kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe residual-after-attn IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. Residual connection after FFN bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_residual_after_ffn_ibp() {
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("glm_pipe_res_after_ffn");
    let input = b.add_input("x", &shape);

    // RMSNorm -> SwiGLU FFN -> add(input, ...)
    let eps = b.add_input("eps", &[1]);
    let nw = b.add_input("nw", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, nw, &shape);

    let ffn_out = add_swiglu_ffn(&mut b, normed, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "ffn");
    let out = b.add_binary_add(input, ffn_out, &shape);
    let def = b.build(out).expect("valid residual-after-FFN kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones(&[HIDDEN_DIM]),
    ];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe residual-after-FFN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 8. Single decoder block (attention + FFN + residuals) bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_single_block_ibp() {
    let mut b = TensorBlockBuilder::new("glm_pipe_single_block");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_decoder_block(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");
    let def = b.build(out).expect("valid single decoder block kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe single decoder block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 9. Two-block decoder stack composition (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_two_block_stack_ibp() {
    let mut b = TensorBlockBuilder::new("glm_pipe_2block_stack");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mid = add_decoder_block(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");
    let out = add_decoder_block(&mut b, mid, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk1");
    let def = b.build(out).expect("valid 2-block decoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe 2-block decoder stack IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. KV cache integration bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_kv_cache_integration_ibp() {
    // KV cache: concatenate cached K/V with new K/V, then run attention.
    // cached_kv: [CACHE_LEN, HIDDEN_DIM] (constant), new_kv: from current input.
    let cache_len = 2;
    let total_len = SEQ_LEN + cache_len;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("glm_pipe_kv_cache");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let cached_k = b.add_input("cached_k", &[cache_len, HIDDEN_DIM]);
    let cached_v = b.add_input("cached_v", &[cache_len, HIDDEN_DIM]);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let k_new = b.add_linear(input, k_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let v_new = b.add_linear(input, v_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Concatenate cached + new K/V along sequence dimension
    let k_full = b.add_concat(&[cached_k, k_new], 0, &[total_len, HIDDEN_DIM]);
    let v_full = b.add_concat(&[cached_v, v_new], 0, &[total_len, HIDDEN_DIM]);

    // Attention: Q [SEQ_LEN, D] @ K_full [total_len, D] -> [SEQ_LEN, D]
    let attn = b.add_attention(
        q,
        k_full,
        v_full,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, HIDDEN_DIM],
    );
    let out = b.add_linear(attn, o_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid KV cache kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[cache_len, HIDDEN_DIM]),
            0.5f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[cache_len, HIDDEN_DIM]),
            0.5f32,
        )),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe KV cache integration IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 11. Final RMSNorm before LM head bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_final_rmsnorm_ibp() {
    // After all decoder blocks, apply final RMSNorm before LM head.
    let mut b = TensorBlockBuilder::new("glm_pipe_final_rmsnorm");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_decoder_block(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");

    // Final RMSNorm
    let eps = b.add_input("final_eps", &[1]);
    let w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(out, eps, 1, w, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(normed).expect("valid final RMSNorm kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe final RMSNorm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 12. Linear LM head logit projection bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_lm_head_projection_ibp() {
    // LM head: Linear(HIDDEN_DIM -> VOCAB_SIZE) producing logits
    let mut b = TensorBlockBuilder::new("glm_pipe_lm_head");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let out = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid LM head kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe LM head logit IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 13. Full GLM decoder block pipeline (CROWN)
// ===========================================================================

#[test]
fn test_glm_decoder_full_block_pipeline_crown() {
    let mut b = TensorBlockBuilder::new("glm_pipe_full_block_crown");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_decoder_block(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");
    let def = b.build(out).expect("valid full decoder block kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM pipe full block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM pipe full block CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 14. Embedding layer output bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_embedding_output_ibp() {
    // Token embedding: gather rows from embedding table [VOCAB_SIZE, HIDDEN_DIM]
    // For verification, model as a linear projection from one-hot-like input
    let mut b = TensorBlockBuilder::new("glm_pipe_embedding");
    let input = b.add_input("token_ids", &[SEQ_LEN, VOCAB_SIZE]);
    let embed_w = b.add_input("embed_w", &[HIDDEN_DIM, VOCAB_SIZE]);
    let out = b.add_linear(input, embed_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid embedding kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // One-hot-like input: each position selects one token
    let input = uniform_bounds(&[SEQ_LEN, VOCAB_SIZE], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe embedding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 15. Position encoding combination bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_position_encoding_ibp() {
    // Embedding + positional encoding addition
    let mut b = TensorBlockBuilder::new("glm_pipe_pos_encode");
    let embed = b.add_input("embed", &[SEQ_LEN, HIDDEN_DIM]);
    let pos_enc = b.add_input("pos_enc", &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_binary_add(embed, pos_enc, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid position encoding kernel");

    let pe_data = rope_cos_sin(SEQ_LEN, HIDDEN_DIM);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe position encoding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // PE bounded in [-1, 1], so output should be bounded around [-2, 2]
    assert!(lo_min > -3.0, "position-encoded lower should be reasonable");
    assert!(hi_max < 3.0, "position-encoded upper should be reasonable");
}

// ===========================================================================
// 16. Multi-block depth composition bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_multiblock_depth_ibp() {
    // Compare bounds after 1, 2, and 4 decoder blocks to observe growth.
    let build_n_blocks = |n: usize| -> BoundedTensor {
        let mut b = TensorBlockBuilder::new(&format!("glm_pipe_depth_{n}"));
        let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
        let mut x = input;
        for i in 0..n {
            x = add_decoder_block(&mut b, x, SEQ_LEN, HIDDEN_DIM, FFN_DIM, &format!("blk{i}"));
        }
        let def = b.build(x).expect("valid n-block decoder");
        let mut bindings = vec![TensorParamBinding::Variable];
        for _ in 0..n {
            push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
        }
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);
        graph.propagate_ibp(&inp).expect("IBP")
    };

    let out1 = build_n_blocks(1);
    let out2 = build_n_blocks(2);
    let out4 = build_n_blocks(4);
    assert_bounds_valid(&out1);
    assert_bounds_valid(&out2);
    assert_bounds_valid(&out4);

    let (l1, h1) = bounds_min_max(&out1);
    let (l2, h2) = bounds_min_max(&out2);
    let (l4, h4) = bounds_min_max(&out4);
    let w1 = h1 - l1;
    let w2 = h2 - l2;
    let w4 = h4 - l4;

    eprintln!("GLM pipe depth: 1-blk w={w1:.4}, 2-blk w={w2:.4}, 4-blk w={w4:.4}");
    assert!(
        w1.is_finite() && w2.is_finite() && w4.is_finite(),
        "all widths must be finite"
    );
}

// ===========================================================================
// 17. Output logit range after LM head (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_output_logit_range_ibp() {
    // Full pipeline: 1 decoder block -> final RMSNorm -> LM head -> softmax
    let mut b = TensorBlockBuilder::new("glm_pipe_logit_range");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let blk_out = add_decoder_block(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");

    // Final RMSNorm
    let eps = b.add_input("final_eps", &[1]);
    let nw = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(blk_out, eps, 1, nw, &[SEQ_LEN, HIDDEN_DIM]);

    // LM head + softmax
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(probs).expect("valid output logit range kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM]));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM pipe output logit range IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-5,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 18. Temperature-scaled logit bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_decoder_temperature_scaled_logits_ibp() {
    // Temperature scaling: logits / temperature before softmax.
    // Model as: LM head logits -> multiply by (1/T) scalar -> softmax.
    let temperature = 0.7_f32;
    let inv_temp = 1.0 / temperature;

    let mut b = TensorBlockBuilder::new("glm_pipe_temp_scaled");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    // LM head projection
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Temperature scaling: logits * (1/T) via element-wise mul with constant
    let temp_scale = b.add_input("temp_scale", &[SEQ_LEN, VOCAB_SIZE]);
    let scaled = b.add_binary_mul(logits, temp_scale, &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax
    let probs = b.add_softmax(scaled, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b
        .build(probs)
        .expect("valid temperature-scaled logit kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[SEQ_LEN, VOCAB_SIZE]),
            inv_temp,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM pipe temperature-scaled (T={temperature}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-5,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}
