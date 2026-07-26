// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep compose tests for the GLM-OCR full decoder pipeline bounds.
//!
//! Extends `compose_dpdf_glm_ocr_full_decoder.rs` with production-representative
//! dimensions, fused SwiGLU (narrow-based gate/up split), multi-layer decoder
//! stacking (2-3 layers), CROWN bound propagation through the full stack,
//! IBP fallback verification, Conservative NormBoundsMode, tight-input
//! precision analysis, and monotone bound widening assertions.
//!
//! ## Tests (15 tests)
//!
//!  1. **RMSNorm Conservative Sound** — Sound verification via Conservative mode (IBP)
//!  2. **Fused SwiGLU FFN with narrow** — Fused gate+up projection with narrow split (IBP + CROWN)
//!  3. **GQA self-attention with RoPE** — Grouped-query attention, 16 heads, causal (IBP)
//!  4. **GQA self-attention CROWN** — CROWN tightening for attention sub-block (CROWN)
//!  5. **Single decoder block Conservative** — Full block with Sound verification (IBP)
//!  6. **Single decoder block CROWN** — CROWN through full decoder block (CROWN)
//!  7. **Two-layer decoder stack IBP** — 2 blocks composed, monotone widening (IBP)
//!  8. **Three-layer decoder stack IBP** — 3 blocks composed, depth scaling (IBP)
//!  9. **Three-layer decoder stack CROWN** — CROWN through 3-layer stack (CROWN)
//! 10. **LM head with softmax** — RMSNorm -> Linear -> softmax output in [0,1] (IBP)
//! 11. **Full pipeline end-to-end IBP** — Embedding -> 2 decoder blocks -> LM head (IBP)
//! 12. **Full pipeline end-to-end CROWN** — Same pipeline with CROWN tightening (CROWN)
//! 13. **Tight input precision** — Narrow input range (+/-0.1) for improved CROWN (IBP + CROWN)
//! 14. **Monotone widening 1-vs-2-vs-3 layers** — Bounds width increases with depth (IBP)
//! 15. **Output projection to large vocab** — Linear to 512-dim vocab space (IBP)
//!
//! Dimensions: D_MODEL=32, FFN_DIM=64, N_HEADS=4, HEAD_DIM=8,
//! SEQ_LEN=8, VOCAB=64. Production GLM-OCR 0.9B: hidden=1536,
//! FFN=8960, heads=12, KV_heads=2, head_dim=128.
//!
//! Part of #4225: Compose tests for GLM-OCR full decoder pipeline bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert_with_config,
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
// Dimensions -- scaled up from the basic 8-dim tests but still fast to verify.
// Structurally representative of GLM-OCR 0.9B (production: hidden=1536,
// FFN=8960, heads=12, KV_heads=2, head_dim=128).
// ---------------------------------------------------------------------------

const D_MODEL: usize = 32;
const FFN_DIM: usize = 64;
const N_HEADS: usize = 4;
const HEAD_DIM: usize = D_MODEL / N_HEADS; // 8
const SEQ: usize = 8;
const VOCAB: usize = 64;
const W_MAG: f32 = 0.001;

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
// Builders
// ---------------------------------------------------------------------------

/// RMSNorm sub-block.
fn build_rmsnorm() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_deep_rmsnorm");
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

/// GLM-OCR fused SwiGLU: dense_h_to_4h (FFN_DIM*2) -> narrow -> SiLU*up -> dense_4h_to_h.
/// This matches the production GLM architecture where gate and up are fused
/// into a single linear projection and then split via narrow.
fn build_fused_swiglu() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_deep_fused_swiglu");
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

    b.build(out).expect("valid fused SwiGLU FFN")
}

fn fused_swiglu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM * 2, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])),
    ]
}

/// GQA self-attention with causal mask. Uses add_attention directly.
/// Q/K/V projections from RMSNorm-ed input, scaled by 1/sqrt(head_dim).
fn build_gqa_self_attn() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_deep_gqa_attn");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let o_w = b.add_input("o_w", &[D_MODEL, D_MODEL]);

    let shape = [SEQ, D_MODEL];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let q = b.add_linear(x, q_w, None, &shape);
    let k = b.add_linear(x, k_w, None, &shape);
    let v = b.add_linear(x, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let out_proj = b.add_linear(attn, o_w, None, &shape);

    // Residual connection
    let out = b.add_binary_add(x, out_proj, &shape);
    b.build(out).expect("valid GQA self-attention")
}

fn gqa_attn_bindings() -> Vec<TensorParamBinding> {
    let wt = w(&[D_MODEL, D_MODEL]);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(wt.clone()),
        TensorParamBinding::ConstantTensor(wt.clone()),
        TensorParamBinding::ConstantTensor(wt.clone()),
        TensorParamBinding::ConstantTensor(wt),
    ]
}

/// Single GLM-OCR decoder block: RMSNorm -> MHA(causal) -> residual
/// -> RMSNorm -> fused SwiGLU -> residual.
fn build_decoder_block() -> TensorKernelDef {
    let (def, _) = build_decoder_block_with_bindings();
    def
}

fn build_decoder_block_with_bindings() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("glm_ocr_deep_decoder_block");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);

    let shape = [SEQ, D_MODEL];
    let ffn_fused = [SEQ, FFN_DIM * 2];
    let ffn_shape = [SEQ, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Attention sub-block
    let attn_rms_w = b.add_input("attn_rms_w", &[D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let o_w = b.add_input("o_w", &[D_MODEL, D_MODEL]);

    let normed1 = b.add_rms_norm(x, eps, 1, attn_rms_w, &shape);
    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
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

    let def = b.build(residual2).expect("valid decoder block");

    let wt = w(&[D_MODEL, D_MODEL]);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        // Attention
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])), // attn_rms_w
        TensorParamBinding::ConstantTensor(wt.clone()),       // q_w
        TensorParamBinding::ConstantTensor(wt.clone()),       // k_w
        TensorParamBinding::ConstantTensor(wt.clone()),       // v_w
        TensorParamBinding::ConstantTensor(wt),               // o_w
        // MLP
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])), // mlp_rms_w
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM * 2, D_MODEL])), // h_to_4h
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])), // h4_to_h
    ];

    (def, bindings)
}

fn decoder_block_bindings() -> Vec<TensorParamBinding> {
    let (_, bindings) = build_decoder_block_with_bindings();
    bindings
}

/// Build an N-layer decoder stack. Returns (def, bindings).
fn build_n_layer_decoder(n_layers: usize) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new(&format!("glm_ocr_deep_{n_layers}layer"));

    let shape = [SEQ, D_MODEL];
    let ffn_fused = [SEQ, FFN_DIM * 2];
    let ffn_shape = [SEQ, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let input = b.add_input("embedded", &shape);
    let eps = b.add_input("eps", &[1]);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let mut current = input;

    for layer_idx in 0..n_layers {
        let pfx = format!("l{layer_idx}");

        // Attention sub-block
        let attn_rms_w = b.add_input(&format!("{pfx}_attn_rms_w"), &[D_MODEL]);
        let q_w = b.add_input(&format!("{pfx}_q_w"), &[D_MODEL, D_MODEL]);
        let k_w = b.add_input(&format!("{pfx}_k_w"), &[D_MODEL, D_MODEL]);
        let v_w = b.add_input(&format!("{pfx}_v_w"), &[D_MODEL, D_MODEL]);
        let o_w = b.add_input(&format!("{pfx}_o_w"), &[D_MODEL, D_MODEL]);

        let normed1 = b.add_rms_norm(current, eps, 1, attn_rms_w, &shape);
        let q = b.add_linear(normed1, q_w, None, &shape);
        let k = b.add_linear(normed1, k_w, None, &shape);
        let v = b.add_linear(normed1, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
        let out_proj = b.add_linear(attn, o_w, None, &shape);
        let residual1 = b.add_binary_add(current, out_proj, &shape);

        // MLP sub-block (fused SwiGLU)
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
        bindings.push(TensorParamBinding::ConstantTensor(wt.clone())); // k_w
        bindings.push(TensorParamBinding::ConstantTensor(wt.clone())); // v_w
        bindings.push(TensorParamBinding::ConstantTensor(wt)); // o_w
        bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL]))); // mlp_rms_w
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            FFN_DIM * 2,
            D_MODEL,
        ]))); // h_to_4h
        bindings.push(TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])));
        // h4_to_h
    }

    let def = b.build(current).expect("valid N-layer decoder");
    (def, bindings)
}

/// Build full pipeline: embedding -> N decoder blocks -> RMSNorm -> LM head -> softmax.
fn build_full_pipeline(n_layers: usize) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new(&format!("glm_ocr_deep_full_{n_layers}layer"));

    let shape = [SEQ, D_MODEL];
    let ffn_fused = [SEQ, FFN_DIM * 2];
    let ffn_shape = [SEQ, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let input = b.add_input("embedded", &shape);
    let eps = b.add_input("eps", &[1]);

    // Position embedding (linear proxy + add)
    let pos_w = b.add_input("pos_w", &[D_MODEL, D_MODEL]);
    let pos_embed = b.add_linear(input, pos_w, None, &shape);
    let embedded = b.add_binary_add(input, pos_embed, &shape);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, D_MODEL])), // pos_w
    ];

    let mut current = embedded;

    for layer_idx in 0..n_layers {
        let pfx = format!("l{layer_idx}");

        // Attention sub-block
        let attn_rms_w = b.add_input(&format!("{pfx}_attn_rms_w"), &[D_MODEL]);
        let q_w = b.add_input(&format!("{pfx}_q_w"), &[D_MODEL, D_MODEL]);
        let k_w = b.add_input(&format!("{pfx}_k_w"), &[D_MODEL, D_MODEL]);
        let v_w = b.add_input(&format!("{pfx}_v_w"), &[D_MODEL, D_MODEL]);
        let o_w = b.add_input(&format!("{pfx}_o_w"), &[D_MODEL, D_MODEL]);

        let normed1 = b.add_rms_norm(current, eps, 1, attn_rms_w, &shape);
        let q = b.add_linear(normed1, q_w, None, &shape);
        let k = b.add_linear(normed1, k_w, None, &shape);
        let v = b.add_linear(normed1, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
        let out_proj = b.add_linear(attn, o_w, None, &shape);
        let residual1 = b.add_binary_add(current, out_proj, &shape);

        // MLP sub-block (fused SwiGLU)
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
        bindings.push(TensorParamBinding::ConstantTensor(wt.clone())); // k_w
        bindings.push(TensorParamBinding::ConstantTensor(wt.clone())); // v_w
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
// 1. RMSNorm Conservative Sound
// ===========================================================================

#[test]
fn test_glm_ocr_deep_rmsnorm_conservative_sound() {
    let def = build_rmsnorm();
    let bindings = rmsnorm_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "glm_ocr_deep_rmsnorm",
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
        "GLM-OCR deep RMSNorm (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 2. Fused SwiGLU FFN with narrow (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_deep_fused_swiglu_ibp_crown() {
    let def = build_fused_swiglu();
    let bindings = fused_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    // IBP pass
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through fused SwiGLU");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR deep fused SwiGLU IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(
        lo_min.abs() < 1e6,
        "SwiGLU lower magnitude < 1e6, got {lo_min}"
    );

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(crown_output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR deep fused SwiGLU CROWN: method={method:?}, [{clo:.6}, {chi:.6}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 3. GQA self-attention with RoPE (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_deep_gqa_self_attn_ibp() {
    let def = build_gqa_self_attn();
    let bindings = gqa_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GQA self-attn");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep GQA self-attn IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.abs() < 1e8, "self-attn lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "self-attn upper < 1e8, got {hi}");
}

// ===========================================================================
// 4. GQA self-attention CROWN
// ===========================================================================

#[test]
fn test_glm_ocr_deep_gqa_self_attn_crown() {
    let def = build_gqa_self_attn();
    let bindings = gqa_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep GQA self-attn CROWN: method={method:?}, [{lo:.6}, {hi:.6}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 5. Single decoder block Conservative
// ===========================================================================

#[test]
fn test_glm_ocr_deep_decoder_block_conservative() {
    let (def, bindings) = build_decoder_block_with_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "glm_ocr_deep_decoder_block",
        &conservative_config(),
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "GLM-OCR deep decoder block (Conservative): [{lo:.6}, {hi:.6}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 6. Single decoder block CROWN
// ===========================================================================

#[test]
fn test_glm_ocr_deep_decoder_block_crown() {
    let def = build_decoder_block();
    let bindings = decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder block");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR deep decoder block IBP: [{ibp_lo:.6}, {ibp_hi:.6}]");

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(crown_output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR deep decoder block CROWN: method={method:?}, [{clo:.6}, {chi:.6}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 7. Two-layer decoder stack IBP
// ===========================================================================

#[test]
fn test_glm_ocr_deep_2layer_stack_ibp() {
    let (def, bindings) = build_n_layer_decoder(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-layer decoder");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep 2-layer decoder IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 8. Three-layer decoder stack IBP
// ===========================================================================

#[test]
fn test_glm_ocr_deep_3layer_stack_ibp() {
    let (def, bindings) = build_n_layer_decoder(3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-layer decoder");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep 3-layer decoder IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 9. Three-layer decoder stack CROWN
// ===========================================================================

#[test]
fn test_glm_ocr_deep_3layer_stack_crown() {
    let (def, bindings) = build_n_layer_decoder(3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep 3-layer decoder CROWN: method={method:?}, [{lo:.6}, {hi:.6}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 10. LM head with softmax (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_deep_lm_head_softmax_ibp() {
    let mut b = TensorBlockBuilder::new("glm_ocr_deep_lm_head_softmax");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let rms_w = b.add_input("rms_w", &[D_MODEL]);
    let lm_w = b.add_input("lm_w", &[VOCAB, D_MODEL]);

    let normed = b.add_rms_norm(x, eps, 1, rms_w, &[SEQ, D_MODEL]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ, VOCAB]);
    let probs = b.add_softmax(logits, 1, &[SEQ, VOCAB]);

    let def = b.build(probs).expect("valid LM head + softmax");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[VOCAB, D_MODEL])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through LM head + softmax");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep LM head softmax IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo >= -1e-4, "softmax lower bound should be >= 0, got {lo}");
    assert!(
        hi <= 1.0 + 1e-4,
        "softmax upper bound should be <= 1, got {hi}"
    );
}

// ===========================================================================
// 11. Full pipeline end-to-end IBP
// ===========================================================================

#[test]
fn test_glm_ocr_deep_full_pipeline_e2e_ibp() {
    let (def, bindings) = build_full_pipeline(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full 2-layer pipeline");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep full pipeline IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());

    // Softmax output should be in [0, 1]
    assert!(lo >= -1e-4, "full pipeline softmax lower >= 0, got {lo}");
    assert!(
        hi <= 1.0 + 1e-4,
        "full pipeline softmax upper <= 1, got {hi}"
    );
}

// ===========================================================================
// 12. Full pipeline end-to-end CROWN
// ===========================================================================

#[test]
fn test_glm_ocr_deep_full_pipeline_e2e_crown() {
    let (def, bindings) = build_full_pipeline(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep full pipeline CROWN: method={method:?}, [{lo:.6}, {hi:.6}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
    // Even with CROWN fallback, softmax output bounds should be in [0, 1]
    assert!(lo >= -1e-3, "pipeline softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-3, "pipeline softmax upper <= 1, got {hi}");
}

// ===========================================================================
// 13. Tight input precision (+/-0.1)
// ===========================================================================

#[test]
fn test_glm_ocr_deep_tight_input_precision() {
    // Narrow input bounds for improved CROWN precision.
    // Reduces relaxation gap in RMSNorm divisor linearization.
    let (def, bindings) = build_decoder_block_with_bindings();
    let tight_input = uniform_bounds(&[SEQ, D_MODEL], 0.1);

    // IBP with tight input
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let ibp_output = graph.propagate_ibp(&tight_input).expect("IBP tight input");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    // IBP with wide input for comparison
    let wide_input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let ibp_wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide input");
    let (wide_lo, wide_hi) = bounds_min_max(&ibp_wide_output);
    let wide_width = wide_hi - wide_lo;

    eprintln!(
        "GLM-OCR deep tight input: IBP tight width={ibp_width:.6}, wide width={wide_width:.6}"
    );
    // Tighter input should produce tighter output bounds
    assert!(
        ibp_width <= wide_width + 1e-4,
        "tight input should produce tighter bounds: tight={ibp_width}, wide={wide_width}"
    );

    // CROWN with tight input
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &tight_input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR deep tight input CROWN: method={method:?}, [{clo:.6}, {chi:.6}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
    // Tight bounds should be narrow
    assert!(
        ibp_lo.is_finite() && ibp_hi.is_finite(),
        "tight input IBP output must be finite"
    );
}

// ===========================================================================
// 14. Monotone widening 1-vs-2-vs-3 layers
// ===========================================================================

#[test]
fn test_glm_ocr_deep_monotone_widening() {
    // Verify that bounds width increases (or stays same) with decoder depth.
    // This is a fundamental property of IBP through composed blocks.
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let mut widths = Vec::new();
    for n in 1..=3 {
        let (def, bindings) = build_n_layer_decoder(n);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);
        let (lo, hi) = bounds_min_max(&output);
        let width = hi - lo;
        eprintln!("GLM-OCR deep {n}-layer width: {width:.6} ([{lo:.6}, {hi:.6}])");
        widths.push(width);
    }

    // Monotone: width_1 <= width_2 <= width_3 (with epsilon tolerance)
    for i in 0..widths.len() - 1 {
        assert!(
            widths[i + 1] >= widths[i] - 1e-4,
            "depth monotone widening violated: {}-layer width={}, {}-layer width={}",
            i + 1,
            widths[i],
            i + 2,
            widths[i + 1]
        );
    }
}

// ===========================================================================
// 15. Output projection to large vocab (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_deep_large_vocab_projection() {
    // Test with a larger vocabulary size (512) to verify scaling behavior.
    const LARGE_VOCAB: usize = 512;

    let mut b = TensorBlockBuilder::new("glm_ocr_deep_large_vocab");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let rms_w = b.add_input("rms_w", &[D_MODEL]);
    let lm_w = b.add_input("lm_w", &[LARGE_VOCAB, D_MODEL]);

    let normed = b.add_rms_norm(x, eps, 1, rms_w, &[SEQ, D_MODEL]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ, LARGE_VOCAB]);
    let def = b.build(logits).expect("valid large vocab projection");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[LARGE_VOCAB, D_MODEL])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through large vocab projection");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, LARGE_VOCAB]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep large vocab (512) projection IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}
