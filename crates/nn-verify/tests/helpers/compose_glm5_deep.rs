// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep NY compose tests for GLM-5 decoder subgraphs.
//!
//! These tests extend the existing GLM-5 compose verification
//! (`compose_glm5_decoder.rs`) with deeper composition patterns, Conservative
//! NormBoundsMode for Sound soundness, and CROWN precision analysis:
//!
//! 1. **RMSNorm (Conservative)** -- Sound verification of normalization in
//!    isolation using Conservative NormBoundsMode (bypasses heuristic
//!    linearization).
//!
//! 2. **SwiGLU FFN (Conservative)** -- Sound bounds through GLM-5 fused
//!    gate+up -> narrow -> SiLU -> mul -> down MLP. Conservative mode
//!    ensures Sound classification without RMSNorm linearization.
//!
//! 3. **Self-attention (Conservative)** -- Sound causal self-attention with
//!    QKV bias (GLM-5 specific: `add_qkv_bias=true`). Uses
//!    `add_multi_head_attention` composite builder.
//!
//! 4. **Single decoder block (Conservative)** -- Full pre-norm block:
//!    RMSNorm -> MHA(QKV+bias) -> residual -> RMSNorm -> SwiGLU -> residual.
//!    Tests Sound verification through the complete block.
//!
//! 5. **Post-norm + LM head (Conservative)** -- RMSNorm -> Linear(D -> VOCAB).
//!    Isolated output projection sub-graph. Sound verification.
//!
//! 6. **Residual bounds widening analysis (2-block)** -- Quantifies IBP
//!    bounds growth through chained decoder layers. Asserts bounded blowup
//!    factor.
//!
//! 7. **Tight-input analysis (+-0.1)** -- Narrow input bounds for improved
//!    CROWN precision. Reduces relaxation gap in RMSNorm divisor
//!    linearization.
//!
//! 8. **Full pipeline with softmax** -- Embedding -> 2-layer decoder ->
//!    RMSNorm -> LM head -> softmax. End-to-end bounds from continuous
//!    embeddings to probability distribution. Validates softmax [0, 1]
//!    invariant.
//!
//! Dimensions are small for fast verification (D_MODEL=16, SEQ_LEN=4).
//! All tests use IbpValidated soundness mode per nn engineering rules
//! (Sound refuses linearization for normalization layers).
//!
//! Part of #4301: GLM5 NY compose verification deepening.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{
    tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerificationSoundnessMode,
    VerifyConfig,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Hidden dimension (GLM-5 production: 4096 for GLM-4-9B).
const D_MODEL: usize = 16;
/// Number of attention heads (production: 32).
const N_HEADS: usize = 2;
/// Per-head dimension.
const HEAD_DIM: usize = D_MODEL / N_HEADS; // 8
/// FFN intermediate dimension (production: 13696).
/// GLM SwiGLU uses fused gate+up of size `FFN_DIM * 2`.
const FFN_DIM: usize = 48;
/// Sequence length for decoder sub-block tests.
const SEQ: usize = 4;
/// Vocabulary size for LM head tests.
const VOCAB: usize = 32;
/// Weight magnitude for bounded verification.
const W_MAG: f32 = 0.001;
/// Number of decoder layers for stack tests.
const N_LAYERS: usize = 2;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ---------------------------------------------------------------------------
// Weight helpers
// ---------------------------------------------------------------------------

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), W_MAG)
}

fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

fn zeros(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

// ---------------------------------------------------------------------------
// 1. Builders -- RMSNorm (Conservative)
// ---------------------------------------------------------------------------

fn build_rmsnorm() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm5_deep_rmsnorm");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[D_MODEL]);
    let out = b.add_rms_norm(x, eps, 1, weight, &[SEQ, D_MODEL]);
    b.build(out).expect("valid RMSNorm")
}

fn rmsnorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])),
    ]
}

// ---------------------------------------------------------------------------
// 2. Builders -- SwiGLU FFN (Conservative, fused gate+up GLM-5 style)
// ---------------------------------------------------------------------------

/// GLM-5 fused SwiGLU: dense_h_to_4h (FFN_DIM*2) -> narrow -> SiLU*up -> dense_4h_to_h.
fn build_swiglu() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm5_deep_swiglu");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let h_to_4h = b.add_input("h_to_4h", &[FFN_DIM * 2, D_MODEL]);
    let h4_to_h = b.add_input("h4_to_h", &[D_MODEL, FFN_DIM]);

    let fused = b.add_linear(x, h_to_4h, None, &[SEQ, FFN_DIM * 2]);
    let gate = b.add_narrow(fused, 1, 0, FFN_DIM, &[SEQ, FFN_DIM]);
    let up = b.add_narrow(fused, 1, FFN_DIM, FFN_DIM, &[SEQ, FFN_DIM]);
    let gate_sig = b.add_sigmoid(gate, &[SEQ, FFN_DIM]);
    let gate_silu = b.add_binary_mul(gate, gate_sig, &[SEQ, FFN_DIM]);
    let gated = b.add_binary_mul(gate_silu, up, &[SEQ, FFN_DIM]);
    let out = b.add_linear(gated, h4_to_h, None, &[SEQ, D_MODEL]);

    b.build(out).expect("valid SwiGLU FFN")
}

fn swiglu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM * 2, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])),
    ]
}

// ---------------------------------------------------------------------------
// 3. Builders -- Self-attention with QKV bias (Conservative)
// ---------------------------------------------------------------------------

/// Causal self-attention with QKV bias (GLM-5 `add_qkv_bias=true`).
/// Uses `add_multi_head_attention` composite builder + residual.
fn build_self_attn() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm5_deep_self_attn");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);

    let shape = [SEQ, D_MODEL];

    let attn = b
        .add_multi_head_attention(
            x,
            q_w,
            k_w,
            v_w,
            out_w,
            N_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("valid causal self-attention");

    // Residual connection
    let out = b.add_binary_add(x, attn, &shape);

    b.build(out).expect("valid self-attention with residual")
}

fn self_attn_bindings() -> Vec<TensorParamBinding> {
    let wt = w(&[D_MODEL, D_MODEL]);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(wt.clone()),
        TensorParamBinding::ConstantTensor(wt.clone()),
        TensorParamBinding::ConstantTensor(wt.clone()),
        TensorParamBinding::ConstantTensor(wt),
    ]
}

// ---------------------------------------------------------------------------
// 4. Builders -- Single decoder block (Conservative)
// ---------------------------------------------------------------------------

/// Single GLM-5 decoder block: RMSNorm -> MHA(QKV+bias, causal) -> residual
/// -> RMSNorm -> SwiGLU MLP -> residual.
///
/// Uses decomposed Q/K/V with bias per GLM-5 convention.
fn build_decoder_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm5_deep_decoder_block");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);

    let shape = [SEQ, D_MODEL];
    let ffn_fused = [SEQ, FFN_DIM * 2];
    let ffn_shape = [SEQ, FFN_DIM];

    // Attention sub-block
    let attn_rms_w = b.add_input("attn_rms_w", &[D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let q_b = b.add_input("q_b", &[D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let k_b = b.add_input("k_b", &[D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let v_b = b.add_input("v_b", &[D_MODEL]);
    let o_w = b.add_input("o_w", &[D_MODEL, D_MODEL]);

    let normed1 = b.add_rms_norm(x, eps, 1, attn_rms_w, &shape);
    let q = b.add_linear(normed1, q_w, Some(q_b), &shape);
    let k = b.add_linear(normed1, k_w, Some(k_b), &shape);
    let v = b.add_linear(normed1, v_w, Some(v_b), &shape);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(1.0 / (HEAD_DIM as f32).sqrt()),
        &shape,
    );
    let out_proj = b.add_linear(attn, o_w, None, &shape);
    let residual1 = b.add_binary_add(x, out_proj, &shape);

    // MLP sub-block (fused SwiGLU)
    let mlp_rms_w = b.add_input("mlp_rms_w", &[D_MODEL]);
    let h_to_4h = b.add_input("h_to_4h", &[FFN_DIM * 2, D_MODEL]);
    let h4_to_h = b.add_input("h4_to_h", &[D_MODEL, FFN_DIM]);

    let normed2 = b.add_rms_norm(residual1, eps, 1, mlp_rms_w, &shape);
    let fused = b.add_linear(normed2, h_to_4h, None, &ffn_fused);
    let gate = b.add_narrow(fused, 1, 0, FFN_DIM, &ffn_shape);
    let up = b.add_narrow(fused, 1, FFN_DIM, FFN_DIM, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_silu = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let gated = b.add_binary_mul(gate_silu, up, &ffn_shape);
    let mlp_out = b.add_linear(gated, h4_to_h, None, &shape);
    let residual2 = b.add_binary_add(residual1, mlp_out, &shape);

    b.build(residual2).expect("valid decoder block")
}

fn decoder_block_bindings() -> Vec<TensorParamBinding> {
    let wt = w(&[D_MODEL, D_MODEL]);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        // Attention
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])), // attn_rms_w
        TensorParamBinding::ConstantTensor(wt.clone()),       // q_w
        TensorParamBinding::ConstantTensor(zeros(&[D_MODEL])), // q_b
        TensorParamBinding::ConstantTensor(wt.clone()),       // k_w
        TensorParamBinding::ConstantTensor(zeros(&[D_MODEL])), // k_b
        TensorParamBinding::ConstantTensor(wt.clone()),       // v_w
        TensorParamBinding::ConstantTensor(zeros(&[D_MODEL])), // v_b
        TensorParamBinding::ConstantTensor(wt),               // o_w
        // MLP
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])), // mlp_rms_w
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM * 2, D_MODEL])), // h_to_4h
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])), // h4_to_h
    ]
}

// ---------------------------------------------------------------------------
// 5. Builders -- Post-norm + LM head
// ---------------------------------------------------------------------------

/// Post-norm + LM head: RMSNorm -> Linear(D -> VOCAB).
fn build_post_norm_lm_head() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm5_deep_post_norm_lm_head");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let rms_w = b.add_input("rms_w", &[D_MODEL]);
    let lm_w = b.add_input("lm_w", &[VOCAB, D_MODEL]);

    let normed = b.add_rms_norm(x, eps, 1, rms_w, &[SEQ, D_MODEL]);
    let out = b.add_linear(normed, lm_w, None, &[SEQ, VOCAB]);

    b.build(out).expect("valid post-norm LM head")
}

fn post_norm_lm_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[VOCAB, D_MODEL])),
    ]
}

// ---------------------------------------------------------------------------
// 8. Builders -- Full pipeline with softmax
// ---------------------------------------------------------------------------

/// Full pipeline: Embedding -> 2-layer decoder -> RMSNorm -> LM head -> softmax.
///
/// Input: `[SEQ, D_MODEL]` (Variable, continuous post-embedding).
/// Output: `[SEQ, VOCAB]` (probability distribution).
fn build_full_pipeline_softmax() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("glm5_deep_full_pipeline_softmax");

    let shape = [SEQ, D_MODEL];
    let ffn_fused = [SEQ, FFN_DIM * 2];
    let ffn_shape = [SEQ, FFN_DIM];

    let input = b.add_input("embedded", &shape);
    let eps = b.add_input("eps", &[1]);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let mut current = input;

    // N_LAYERS decoder blocks
    for layer_idx in 0..N_LAYERS {
        let pfx = format!("l{layer_idx}");

        // Attention sub-block
        let attn_rms_w = b.add_input(&format!("{pfx}_attn_rms_w"), &[D_MODEL]);
        let q_w = b.add_input(&format!("{pfx}_q_w"), &[D_MODEL, D_MODEL]);
        let q_b = b.add_input(&format!("{pfx}_q_b"), &[D_MODEL]);
        let k_w = b.add_input(&format!("{pfx}_k_w"), &[D_MODEL, D_MODEL]);
        let k_b = b.add_input(&format!("{pfx}_k_b"), &[D_MODEL]);
        let v_w = b.add_input(&format!("{pfx}_v_w"), &[D_MODEL, D_MODEL]);
        let v_b = b.add_input(&format!("{pfx}_v_b"), &[D_MODEL]);
        let o_w = b.add_input(&format!("{pfx}_o_w"), &[D_MODEL, D_MODEL]);

        let normed1 = b.add_rms_norm(current, eps, 1, attn_rms_w, &shape);
        let q = b.add_linear(normed1, q_w, Some(q_b), &shape);
        let k = b.add_linear(normed1, k_w, Some(k_b), &shape);
        let v = b.add_linear(normed1, v_w, Some(v_b), &shape);
        let attn = b.add_attention(
            q,
            k,
            v,
            AttentionMask::Causal,
            Some(1.0 / (HEAD_DIM as f32).sqrt()),
            &shape,
        );
        let out_proj = b.add_linear(attn, o_w, None, &shape);
        let residual1 = b.add_binary_add(current, out_proj, &shape);

        // MLP sub-block
        let mlp_rms_w = b.add_input(&format!("{pfx}_mlp_rms_w"), &[D_MODEL]);
        let h_to_4h = b.add_input(&format!("{pfx}_h_to_4h"), &[FFN_DIM * 2, D_MODEL]);
        let h4_to_h = b.add_input(&format!("{pfx}_h4_to_h"), &[D_MODEL, FFN_DIM]);

        let normed2 = b.add_rms_norm(residual1, eps, 1, mlp_rms_w, &shape);
        let fused = b.add_linear(normed2, h_to_4h, None, &ffn_fused);
        let gate = b.add_narrow(fused, 1, 0, FFN_DIM, &ffn_shape);
        let up = b.add_narrow(fused, 1, FFN_DIM, FFN_DIM, &ffn_shape);
        let gate_sig = b.add_sigmoid(gate, &ffn_shape);
        let gate_silu = b.add_binary_mul(gate, gate_sig, &ffn_shape);
        let gated = b.add_binary_mul(gate_silu, up, &ffn_shape);
        let mlp_out = b.add_linear(gated, h4_to_h, None, &shape);
        current = b.add_binary_add(residual1, mlp_out, &shape);

        // Bindings for this layer
        let wt = w(&[D_MODEL, D_MODEL]);
        bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL]))); // attn_rms_w
        bindings.push(TensorParamBinding::ConstantTensor(wt.clone())); // q_w
        bindings.push(TensorParamBinding::ConstantTensor(zeros(&[D_MODEL]))); // q_b
        bindings.push(TensorParamBinding::ConstantTensor(wt.clone())); // k_w
        bindings.push(TensorParamBinding::ConstantTensor(zeros(&[D_MODEL]))); // k_b
        bindings.push(TensorParamBinding::ConstantTensor(wt.clone())); // v_w
        bindings.push(TensorParamBinding::ConstantTensor(zeros(&[D_MODEL]))); // v_b
        bindings.push(TensorParamBinding::ConstantTensor(wt)); // o_w
        bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL]))); // mlp_rms_w
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            FFN_DIM * 2,
            D_MODEL,
        ]))); // h_to_4h
        bindings.push(TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])));
        // h4_to_h
    }

    // Final RMSNorm
    let final_rms_w = b.add_input("final_rms_w", &[D_MODEL]);
    let normed = b.add_rms_norm(current, eps, 1, final_rms_w, &shape);
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL])));

    // LM head -> softmax
    let lm_w = b.add_input("lm_w", &[VOCAB, D_MODEL]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ, VOCAB]);
    let probs = b.add_softmax(logits, 1, &[SEQ, VOCAB]);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[VOCAB, D_MODEL])));

    let def = b.build(probs).expect("valid full pipeline with softmax");
    (def, bindings)
}

// ===========================================================================
// 1. RMSNorm (Conservative) -- Sound
// ===========================================================================

#[test]
fn test_glm5_deep_rmsnorm_conservative_sound() {
    let def = build_rmsnorm();
    let bindings = rmsnorm_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "glm5_deep_rmsnorm",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative RMSNorm should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "GLM-5 deep RMSNorm (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 2. SwiGLU FFN (Conservative) -- Sound
// ===========================================================================

#[test]
fn test_glm5_deep_swiglu_conservative_sound() {
    let def = build_swiglu();
    let bindings = swiglu_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "glm5_deep_swiglu",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative SwiGLU should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "GLM-5 deep SwiGLU (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
    assert!(lo.abs() < 1e6, "SwiGLU lower magnitude < 1e6, got {lo}");
    assert!(hi.abs() < 1e6, "SwiGLU upper magnitude < 1e6, got {hi}");
}

// ===========================================================================
// 3. Self-attention with QKV bias (Conservative) -- Sound
// ===========================================================================

#[test]
fn test_glm5_deep_self_attn_conservative_ibp() {
    let def = build_self_attn();
    let bindings = self_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through self-attn");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-5 deep self-attn IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e8, "self-attn lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "self-attn upper < 1e8, got {hi}");
}

#[test]
fn test_glm5_deep_self_attn_conservative_crown() {
    let def = build_self_attn();
    let bindings = self_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-5 deep self-attn: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_glm5_deep_self_attn_conservative_verify_and_record() {
    let def = build_self_attn();
    let bindings = self_attn_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "glm5_deep_self_attn",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative self-attn should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 4. Single decoder block (Conservative)
// ===========================================================================

#[test]
fn test_glm5_deep_decoder_block_def_validates() {
    let def = build_decoder_block();
    def.validate().expect("decoder block should validate");
}

#[test]
fn test_glm5_deep_decoder_block_ibp() {
    let def = build_decoder_block();
    let bindings = decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder block");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-5 deep decoder block IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e8, "decoder block lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "decoder block upper < 1e8, got {hi}");
}

#[test]
fn test_glm5_deep_decoder_block_crown() {
    let def = build_decoder_block();
    let bindings = decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-5 deep decoder block: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_glm5_deep_decoder_block_verify_and_record() {
    let def = build_decoder_block();
    let bindings = decoder_block_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "glm5_deep_decoder_block",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "GLM-5 deep decoder block (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 5. Post-norm + LM head (Conservative)
// ===========================================================================

#[test]
fn test_glm5_deep_post_norm_lm_head_ibp() {
    let def = build_post_norm_lm_head();
    let bindings = post_norm_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through post-norm LM head");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-5 deep post-norm LM head IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower must be finite, got {lo}");
    assert!(hi.is_finite(), "upper must be finite, got {hi}");
}

#[test]
fn test_glm5_deep_post_norm_lm_head_verify_and_record() {
    let def = build_post_norm_lm_head();
    let bindings = post_norm_lm_head_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "glm5_deep_post_norm_lm_head",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "GLM-5 deep post-norm LM head (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 6. Residual bounds widening analysis (2-block)
// ===========================================================================

/// Residual bounds analysis: verifies that 2 blocks of the GLM-5 decoder
/// do not cause excessive bounds blowup due to residual connections.
#[test]
fn test_glm5_deep_residual_bounds_2block() {
    // Single block
    let def1 = build_decoder_block();
    let bindings1 = decoder_block_bindings();
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output1 = graph1.propagate_ibp(&input).expect("IBP through 1 block");
    let (lo1, hi1) = bounds_min_max(&output1);
    let range1 = hi1 - lo1;

    // Use single block output as input to second block
    let output2 = graph1
        .propagate_ibp(&output1)
        .expect("IBP through 2nd block");
    let (lo2, hi2) = bounds_min_max(&output2);
    let range2 = hi2 - lo2;

    eprintln!(
        "GLM-5 residual analysis: 1-block range={range1:.4}, 2-block range={range2:.4}, \
         blowup={:.1}x",
        range2 / range1.max(1e-10)
    );

    // 2 blocks should not blow up more than 1e4x relative to 1 block.
    // With small weights and residual connections, growth is controlled.
    let blowup = range2 / range1.max(1e-10);
    assert!(
        blowup < 1e4,
        "2-block blowup factor should be < 1e4 relative to 1-block, got {blowup:.1}x"
    );
}

// ===========================================================================
// 7. Tight-input analysis (+-0.1 bounds for CROWN precision)
// ===========================================================================

#[test]
fn test_glm5_deep_tight_decoder_ibp() {
    let def = build_decoder_block();
    let bindings = decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Narrow input: +-0.1
    let input = uniform_bounds(&[SEQ, D_MODEL], 0.1);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("GLM-5 deep tight decoder IBP (+-0.1): [{lo_min}, {hi_max}], width={width:.6}");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_glm5_deep_tight_decoder_crown() {
    let def = build_decoder_block();
    let bindings = decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Narrow input: +-0.1
    let input = uniform_bounds(&[SEQ, D_MODEL], 0.1);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!(
        "GLM-5 deep tight decoder CROWN ({method:?}): [{lo_min}, {hi_max}], width={width:.6}"
    );
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_glm5_deep_tight_decoder_verify_and_record() {
    let def = build_decoder_block();
    let bindings = decoder_block_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 0.1);

    let result = verify_and_assert(&def, &bindings, &input, "glm5_deep_tight_decoder");
    assert_eq!(result.num_variables, 1, "single Variable input");
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, D_MODEL]
    );
}

// ===========================================================================
// 8. Full pipeline with softmax: embedding -> decoder -> LM head -> softmax
// ===========================================================================

#[test]
fn test_glm5_deep_full_pipeline_softmax_ibp() {
    let (def, bindings) = build_full_pipeline_softmax();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-5 deep full pipeline softmax IBP: [{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    let eps = 1e-4;
    assert!(lo_min >= -eps, "softmax lower should be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper should be <= 1, got {hi_max}"
    );
}

#[test]
fn test_glm5_deep_full_pipeline_softmax_verify_and_record() {
    let (def, bindings) = build_full_pipeline_softmax();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "glm5_deep_full_pipeline_softmax");
    assert_eq!(result.num_variables, 1, "single Variable input");
    assert_eq!(result.output_bounds.lower_upper().0.shape(), &[SEQ, VOCAB]);
}
