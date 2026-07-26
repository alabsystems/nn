// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Granite-Docling encoder-decoder cross-attention compose tests.
//!
//! Extends `compose_dpdf_granite_docling_enc_dec.rs` (15 tests) with deeper
//! cross-attention verification: asymmetric encoder/decoder dimensions, explicit
//! multi-head head-splitting, encoder output re-projection, full encoder ->
//! cross-attention -> decoder pipeline, post-cross-attention layer norm,
//! combined self+cross attention, CROWN tightening, and multi-layer decoder
//! bound accumulation.
//!
//! ## Tests (10 tests)
//!
//! 16. **Cross-attention KQV with asymmetric dimensions** (IBP)
//! 17. **Multi-head cross-attention with explicit head split** (IBP + CROWN)
//! 18. **Encoder output re-projection to decoder dim** (IBP)
//! 19. **Full encoder -> cross-attention -> decoder pipeline** (IBP)
//! 20. **Layer norm after cross-attention** (IBP + CROWN)
//! 21. **Decoder with self-attention + cross-attention** (IBP + CROWN)
//! 22. **CROWN tightening for cross-attention: wide vs narrow** (CROWN)
//! 23. **3-layer decoder stack with accumulating bounds** (IBP)
//! 24. **Cross-attention with varying encoder sequence lengths** (IBP)
//! 25. **Verify-and-record: cross-attention isolation** (IBP)
//!
//! Architecture reference: Granite-Docling-258M (SigLIP2 + Granite decoder)
//! Part of #4228.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- structurally representative, small for fast verification.
// Encoder and decoder have DIFFERENT hidden dimensions to exercise the
// asymmetric cross-attention path (encoder dim != decoder dim).
// ---------------------------------------------------------------------------

/// Encoder (vision) hidden dimension.
const ENC_DIM: usize = 24;
/// Decoder (LM) hidden dimension.
const DEC_DIM: usize = 16;
/// Encoder sequence length (vision patches).
const ENC_SEQ: usize = 6;
/// Decoder sequence length (text tokens).
const DEC_SEQ: usize = 4;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Head dimension derived from decoder: DEC_DIM / NUM_HEADS.
const HEAD_DIM: usize = DEC_DIM / NUM_HEADS; // 4
/// FFN intermediate dimension (decoder).
const FFN_DIM: usize = 32;
/// Weight magnitude for constant tensors.
const W_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn w(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), W_MAG))
}

fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

fn eps_scalar() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

fn const_tensor(shape: &[usize], val: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), val))
}

/// Build asymmetric cross-attention: Q from decoder [DEC_SEQ, DEC_DIM],
/// K/V from encoder [ENC_SEQ, ENC_DIM] projected to DEC_DIM.
fn add_asymmetric_cross_attn(
    b: &mut TensorBlockBuilder,
    dec_input: TensorNodeId,
    enc_input: TensorNodeId,
    prefix: &str,
) -> TensorNodeId {
    let dec_shape = [DEC_SEQ, DEC_DIM];

    // LayerNorm on decoder side
    let ln_w = b.add_input(&format!("{prefix}_ln_w"), &[DEC_DIM]);
    let ln_b = b.add_input(&format!("{prefix}_ln_b"), &[DEC_DIM]);
    let eps = b.add_input(&format!("{prefix}_ln_eps"), &[1]);
    let normed_dec = b.add_layer_norm(dec_input, eps, 1, ln_w, ln_b, &dec_shape);

    // Q from decoder (DEC_DIM -> DEC_DIM)
    let q_w = b.add_input(&format!("{prefix}_q_w"), &[DEC_DIM, DEC_DIM]);
    // K from encoder (ENC_DIM -> DEC_DIM) -- asymmetric projection
    let k_w = b.add_input(&format!("{prefix}_k_w"), &[DEC_DIM, ENC_DIM]);
    // V from encoder (ENC_DIM -> DEC_DIM) -- asymmetric projection
    let v_w = b.add_input(&format!("{prefix}_v_w"), &[DEC_DIM, ENC_DIM]);
    // Output projection (DEC_DIM -> DEC_DIM)
    let o_w = b.add_input(&format!("{prefix}_o_w"), &[DEC_DIM, DEC_DIM]);

    // Manual Q/K/V projections to handle the dimension mismatch
    let q = b.add_linear(normed_dec, q_w, None, &dec_shape);
    let k = b.add_linear(enc_input, k_w, None, &[ENC_SEQ, DEC_DIM]);
    let v = b.add_linear(enc_input, v_w, None, &[ENC_SEQ, DEC_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &dec_shape);
    let attn_out = b.add_linear(attn, o_w, None, &dec_shape);

    // Residual
    b.add_binary_add(dec_input, attn_out, &dec_shape)
}

/// Bindings for asymmetric cross-attention block.
fn asymmetric_cross_attn_bindings() -> Vec<TensorParamBinding> {
    vec![
        ones(&[DEC_DIM]),       // ln_w
        bias_zero(&[DEC_DIM]),  // ln_b
        eps_scalar(),           // ln_eps
        w(&[DEC_DIM, DEC_DIM]), // q_w
        w(&[DEC_DIM, ENC_DIM]), // k_w (asymmetric!)
        w(&[DEC_DIM, ENC_DIM]), // v_w (asymmetric!)
        w(&[DEC_DIM, DEC_DIM]), // o_w
    ]
}

/// Add a SwiGLU FFN block with RMSNorm pre-norm + residual.
fn add_swiglu_ffn(b: &mut TensorBlockBuilder, input: TensorNodeId, prefix: &str) -> TensorNodeId {
    let shape = [DEC_SEQ, DEC_DIM];
    let ffn_shape = [DEC_SEQ, FFN_DIM];

    let rms_eps = b.add_input(&format!("{prefix}_rms_eps"), &[1]);
    let rms_w = b.add_input(&format!("{prefix}_rms_w"), &[DEC_DIM]);
    let normed = b.add_rms_norm(input, rms_eps, 1, rms_w, &shape);

    let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[FFN_DIM, DEC_DIM]);
    let up_w = b.add_input(&format!("{prefix}_up_w"), &[FFN_DIM, DEC_DIM]);
    let down_w = b.add_input(&format!("{prefix}_down_w"), &[DEC_DIM, FFN_DIM]);

    let gate = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    b.add_binary_add(input, ffn_out, &shape)
}

fn swiglu_ffn_bindings() -> Vec<TensorParamBinding> {
    vec![
        eps_scalar(),
        ones(&[DEC_DIM]),
        w(&[FFN_DIM, DEC_DIM]),
        w(&[FFN_DIM, DEC_DIM]),
        w(&[DEC_DIM, FFN_DIM]),
    ]
}

/// Decoder self-attention block with RMSNorm + causal mask + residual.
fn add_self_attn(b: &mut TensorBlockBuilder, input: TensorNodeId, prefix: &str) -> TensorNodeId {
    let shape = [DEC_SEQ, DEC_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let rms_eps = b.add_input(&format!("{prefix}_rms_eps"), &[1]);
    let rms_w = b.add_input(&format!("{prefix}_rms_w"), &[DEC_DIM]);
    let normed = b.add_rms_norm(input, rms_eps, 1, rms_w, &shape);

    let q_w = b.add_input(&format!("{prefix}_q_w"), &[DEC_DIM, DEC_DIM]);
    let k_w = b.add_input(&format!("{prefix}_k_w"), &[DEC_DIM, DEC_DIM]);
    let v_w = b.add_input(&format!("{prefix}_v_w"), &[DEC_DIM, DEC_DIM]);
    let o_w = b.add_input(&format!("{prefix}_o_w"), &[DEC_DIM, DEC_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);

    b.add_binary_add(input, attn_out, &shape)
}

fn self_attn_bindings() -> Vec<TensorParamBinding> {
    vec![
        eps_scalar(),
        ones(&[DEC_DIM]),
        w(&[DEC_DIM, DEC_DIM]),
        w(&[DEC_DIM, DEC_DIM]),
        w(&[DEC_DIM, DEC_DIM]),
        w(&[DEC_DIM, DEC_DIM]),
    ]
}

// ===========================================================================
// 16. Cross-attention KQV with asymmetric dimensions (IBP)
// ===========================================================================

#[test]
fn test_gd_ext_asymmetric_cross_attn_kqv_ibp() {
    let mut b = TensorBlockBuilder::new("gd_ext_asym_xattn_kqv");
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, DEC_DIM]);
    let enc_in = b.add_input("enc_features", &[ENC_SEQ, ENC_DIM]);
    let out = add_asymmetric_cross_attn(&mut b, dec_in, enc_in, "xa0");
    let def = b.build(out).expect("valid asymmetric cross-attn kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[ENC_SEQ, ENC_DIM], 0.5),
    ];
    bindings.extend(asymmetric_cross_attn_bindings());
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, DEC_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, DEC_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD ext asymmetric cross-attn KQV IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 17. Multi-head cross-attention with explicit head split (IBP + CROWN)
// ===========================================================================

/// Uses `add_multi_head_cross_attention` from TensorBlockBuilder to exercise
/// the built-in MHA cross-attention path with head splitting.
fn build_mha_cross_attn_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gd_ext_mha_xattn_head_split");
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, DEC_DIM]);
    let enc_in = b.add_input("enc_features", &[ENC_SEQ, DEC_DIM]);

    let q_w = b.add_input("q_w", &[DEC_DIM, DEC_DIM]);
    let k_w = b.add_input("k_w", &[DEC_DIM, DEC_DIM]);
    let v_w = b.add_input("v_w", &[DEC_DIM, DEC_DIM]);
    let o_w = b.add_input("o_w", &[DEC_DIM, DEC_DIM]);

    let xattn = b
        .add_multi_head_cross_attention(
            dec_in,
            enc_in,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[DEC_SEQ, DEC_DIM],
        )
        .expect("valid MHA cross-attn");

    // Residual
    let out = b.add_binary_add(dec_in, xattn, &[DEC_SEQ, DEC_DIM]);
    let def = b.build(out).expect("valid kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[ENC_SEQ, DEC_DIM], 0.5),
        w(&[DEC_DIM, DEC_DIM]),
        w(&[DEC_DIM, DEC_DIM]),
        w(&[DEC_DIM, DEC_DIM]),
        w(&[DEC_DIM, DEC_DIM]),
    ];
    (def, bindings)
}

#[test]
fn test_gd_ext_mha_cross_attn_head_split_ibp() {
    let (def, bindings) = build_mha_cross_attn_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, DEC_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, DEC_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD ext MHA cross-attn head split IBP: [{lo:.6}, {hi:.6}]");
}

#[test]
fn test_gd_ext_mha_cross_attn_head_split_crown() {
    let (def, bindings) = build_mha_cross_attn_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, DEC_DIM], 0.5);

    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD ext MHA cross-attn CROWN ({method:?}): [{lo:.6}, {hi:.6}]");
}

// ===========================================================================
// 18. Encoder output re-projection to decoder dim (IBP)
// ===========================================================================

#[test]
fn test_gd_ext_encoder_reprojection_ibp() {
    let mut b = TensorBlockBuilder::new("gd_ext_enc_reproj");

    // Encoder features in encoder dim
    let enc_in = b.add_input("enc_features", &[ENC_SEQ, ENC_DIM]);

    // Re-projection: ENC_DIM -> DEC_DIM with bias + LayerNorm
    let proj_w = b.add_input("proj_w", &[DEC_DIM, ENC_DIM]);
    let proj_b = b.add_input("proj_b", &[DEC_DIM]);
    let projected = b.add_linear(enc_in, proj_w, Some(proj_b), &[ENC_SEQ, DEC_DIM]);

    // LayerNorm after projection (common pattern in VLM adapters)
    let ln_w = b.add_input("ln_w", &[DEC_DIM]);
    let ln_b = b.add_input("ln_b", &[DEC_DIM]);
    let eps = b.add_input("ln_eps", &[1]);
    let out = b.add_layer_norm(projected, eps, 1, ln_w, ln_b, &[ENC_SEQ, DEC_DIM]);
    let def = b.build(out).expect("valid encoder reprojection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[DEC_DIM, ENC_DIM]),
        bias_zero(&[DEC_DIM]),
        ones(&[DEC_DIM]),
        bias_zero(&[DEC_DIM]),
        eps_scalar(),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[ENC_SEQ, ENC_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ, DEC_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD ext encoder reprojection IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 19. Full encoder -> cross-attention -> decoder pipeline (IBP)
// ===========================================================================

#[test]
fn test_gd_ext_full_enc_xattn_dec_pipeline_ibp() {
    let mut b = TensorBlockBuilder::new("gd_ext_enc_xattn_dec_pipe");

    // Encoder features (variable input) in ENC_DIM
    let enc_in = b.add_input("enc_features", &[ENC_SEQ, ENC_DIM]);

    // Re-project encoder -> DEC_DIM
    let proj_w = b.add_input("proj_w", &[DEC_DIM, ENC_DIM]);
    let enc_proj = b.add_linear(enc_in, proj_w, None, &[ENC_SEQ, DEC_DIM]);

    // Decoder input (constant, represents text embedding)
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, DEC_DIM]);

    // Cross-attention: decoder queries, encoder K/V (now in DEC_DIM)
    let q_w = b.add_input("xa_q_w", &[DEC_DIM, DEC_DIM]);
    let k_w = b.add_input("xa_k_w", &[DEC_DIM, DEC_DIM]);
    let v_w = b.add_input("xa_v_w", &[DEC_DIM, DEC_DIM]);
    let o_w = b.add_input("xa_o_w", &[DEC_DIM, DEC_DIM]);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let dec_shape = [DEC_SEQ, DEC_DIM];

    let q = b.add_linear(dec_in, q_w, None, &dec_shape);
    let k = b.add_linear(enc_proj, k_w, None, &[ENC_SEQ, DEC_DIM]);
    let v = b.add_linear(enc_proj, v_w, None, &[ENC_SEQ, DEC_DIM]);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &dec_shape);
    let xa_out = b.add_linear(attn, o_w, None, &dec_shape);
    let xa_res = b.add_binary_add(dec_in, xa_out, &dec_shape);

    // SwiGLU FFN after cross-attention
    let out = add_swiglu_ffn(&mut b, xa_res, "ffn0");
    let def = b.build(out).expect("valid enc->xattn->dec pipeline kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,           // enc_features
        w(&[DEC_DIM, ENC_DIM]),                 // proj_w
        const_tensor(&[DEC_SEQ, DEC_DIM], 0.1), // dec_input (constant)
        w(&[DEC_DIM, DEC_DIM]),                 // xa_q_w
        w(&[DEC_DIM, DEC_DIM]),                 // xa_k_w
        w(&[DEC_DIM, DEC_DIM]),                 // xa_v_w
        w(&[DEC_DIM, DEC_DIM]),                 // xa_o_w
    ];
    bindings.extend(swiglu_ffn_bindings());

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[ENC_SEQ, ENC_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, DEC_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD ext enc->xattn->dec pipeline IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 20. Layer norm after cross-attention (IBP + CROWN)
// ===========================================================================

fn build_xattn_layernorm_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gd_ext_xattn_layernorm");
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, DEC_DIM]);
    let enc_in = b.add_input("enc_features", &[ENC_SEQ, ENC_DIM]);

    let xa_out = add_asymmetric_cross_attn(&mut b, dec_in, enc_in, "xa0");

    // Post-cross-attention LayerNorm (common in encoder-decoder transformers)
    let ln_w = b.add_input("post_ln_w", &[DEC_DIM]);
    let ln_b = b.add_input("post_ln_b", &[DEC_DIM]);
    let eps = b.add_input("post_ln_eps", &[1]);
    let out = b.add_layer_norm(xa_out, eps, 1, ln_w, ln_b, &[DEC_SEQ, DEC_DIM]);
    let def = b.build(out).expect("valid xattn+layernorm kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[ENC_SEQ, ENC_DIM], 0.5),
    ];
    bindings.extend(asymmetric_cross_attn_bindings());
    bindings.push(ones(&[DEC_DIM])); // post_ln_w
    bindings.push(bias_zero(&[DEC_DIM])); // post_ln_b
    bindings.push(eps_scalar()); // post_ln_eps

    (def, bindings)
}

#[test]
fn test_gd_ext_xattn_layernorm_ibp() {
    let (def, bindings) = build_xattn_layernorm_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, DEC_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, DEC_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD ext xattn+layernorm IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

#[test]
fn test_gd_ext_xattn_layernorm_crown() {
    let (def, bindings) = build_xattn_layernorm_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, DEC_DIM], 0.5);

    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD ext xattn+layernorm CROWN ({method:?}): [{lo:.6}, {hi:.6}]");
}

// ===========================================================================
// 21. Decoder with self-attention + cross-attention (IBP + CROWN)
// ===========================================================================

fn build_self_plus_cross_attn_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gd_ext_sa_xa_combined");
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, DEC_DIM]);
    let enc_in = b.add_input("enc_features", &[ENC_SEQ, ENC_DIM]);

    // Self-attention first (causal)
    let sa_out = add_self_attn(&mut b, dec_in, "sa0");
    // Cross-attention second (asymmetric encoder -> decoder)
    let xa_out = add_asymmetric_cross_attn(&mut b, sa_out, enc_in, "xa0");
    // SwiGLU FFN
    let out = add_swiglu_ffn(&mut b, xa_out, "ffn0");
    let def = b.build(out).expect("valid sa+xa+ffn kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[ENC_SEQ, ENC_DIM], 0.5),
    ];
    bindings.extend(self_attn_bindings());
    bindings.extend(asymmetric_cross_attn_bindings());
    bindings.extend(swiglu_ffn_bindings());

    (def, bindings)
}

#[test]
fn test_gd_ext_self_plus_cross_attn_ibp() {
    let (def, bindings) = build_self_plus_cross_attn_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, DEC_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, DEC_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD ext self+cross attn IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

#[test]
fn test_gd_ext_self_plus_cross_attn_crown() {
    let (def, bindings) = build_self_plus_cross_attn_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, DEC_DIM], 0.3);

    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD ext self+cross attn CROWN ({method:?}): [{lo:.6}, {hi:.6}]");
}

// ===========================================================================
// 22. CROWN tightening for cross-attention: wide vs narrow (CROWN)
// ===========================================================================

#[test]
fn test_gd_ext_crown_tightening_wide_vs_narrow() {
    let mut b = TensorBlockBuilder::new("gd_ext_crown_tight_xattn");
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, DEC_DIM]);
    let enc_in = b.add_input("enc_features", &[ENC_SEQ, ENC_DIM]);
    let out = add_asymmetric_cross_attn(&mut b, dec_in, enc_in, "xa0");
    let def = b.build(out).expect("valid kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[ENC_SEQ, ENC_DIM], 0.5),
    ];
    bindings.extend(asymmetric_cross_attn_bindings());
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Wide input: +/- 1.0
    let wide_input = uniform_bounds(&[DEC_SEQ, DEC_DIM], 1.0);
    let wide_ibp = graph.propagate_ibp(&wide_input).expect("wide IBP");
    let (wide_lo, wide_hi) = bounds_min_max(&wide_ibp);
    let wide_width = wide_hi - wide_lo;

    // Narrow input: +/- 0.1
    let narrow_input = uniform_bounds(&[DEC_SEQ, DEC_DIM], 0.1);
    let (method, narrow_crown, _) = assert_crown_tighter_when_not_fallback(&graph, &narrow_input);
    let (narrow_lo, narrow_hi) = bounds_min_max(&narrow_crown);
    let narrow_width = narrow_hi - narrow_lo;

    eprintln!(
        "GD ext CROWN tightening: wide IBP width={wide_width:.4}, \
         narrow CROWN ({method:?}) width={narrow_width:.4}"
    );
    eprintln!(
        "  Tightening ratio: {:.2}x",
        wide_width / narrow_width.max(1e-10)
    );

    assert!(wide_width.is_finite() && narrow_width.is_finite());
    // CROWN on narrow input should produce meaningfully tighter bounds
    // than IBP on wide input (unless both are very small).
    assert!(
        narrow_width <= wide_width + 1e-4,
        "narrow CROWN should not exceed wide IBP: {narrow_width} > {wide_width}"
    );
}

// ===========================================================================
// 23. 3-layer decoder stack with accumulating bounds (IBP)
// ===========================================================================

#[test]
fn test_gd_ext_3layer_decoder_stack_ibp() {
    let mut b = TensorBlockBuilder::new("gd_ext_3layer_dec_stack");
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, DEC_DIM]);
    let enc_in = b.add_input("enc_features", &[ENC_SEQ, ENC_DIM]);

    // Layer 0: self-attn -> cross-attn -> FFN
    let sa0 = add_self_attn(&mut b, dec_in, "l0_sa");
    let xa0 = add_asymmetric_cross_attn(&mut b, sa0, enc_in, "l0_xa");
    let l0 = add_swiglu_ffn(&mut b, xa0, "l0_ffn");

    // Layer 1
    let sa1 = add_self_attn(&mut b, l0, "l1_sa");
    let xa1 = add_asymmetric_cross_attn(&mut b, sa1, enc_in, "l1_xa");
    let l1 = add_swiglu_ffn(&mut b, xa1, "l1_ffn");

    // Layer 2
    let sa2 = add_self_attn(&mut b, l1, "l2_sa");
    let xa2 = add_asymmetric_cross_attn(&mut b, sa2, enc_in, "l2_xa");
    let out = add_swiglu_ffn(&mut b, xa2, "l2_ffn");

    let def = b.build(out).expect("valid 3-layer decoder kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[ENC_SEQ, ENC_DIM], 0.5),
    ];
    // 3 layers x (self_attn + cross_attn + ffn)
    for _ in 0..3 {
        bindings.extend(self_attn_bindings());
        bindings.extend(asymmetric_cross_attn_bindings());
        bindings.extend(swiglu_ffn_bindings());
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, DEC_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, DEC_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;
    eprintln!("GD ext 3-layer decoder stack IBP: [{lo:.6}, {hi:.6}], width={width:.4}");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 24. Cross-attention with varying encoder sequence lengths (IBP)
// ===========================================================================

#[test]
fn test_gd_ext_varying_encoder_seq_lengths_ibp() {
    // Compare bounds for different encoder sequence lengths (2 vs 8) to verify
    // that longer encoder sequences (more K/V context) produce different but
    // still valid bounds.
    let dec_shape = [DEC_SEQ, DEC_DIM];

    let build_for_enc_seq = |enc_seq: usize| -> (f32, f32) {
        let mut b = TensorBlockBuilder::new(&format!("gd_ext_enc_seq_{enc_seq}"));
        let dec_in = b.add_input("dec_input", &dec_shape);
        let enc_in = b.add_input("enc_features", &[enc_seq, DEC_DIM]);

        let q_w = b.add_input("q_w", &[DEC_DIM, DEC_DIM]);
        let k_w = b.add_input("k_w", &[DEC_DIM, DEC_DIM]);
        let v_w = b.add_input("v_w", &[DEC_DIM, DEC_DIM]);
        let o_w = b.add_input("o_w", &[DEC_DIM, DEC_DIM]);

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let q = b.add_linear(dec_in, q_w, None, &dec_shape);
        let k = b.add_linear(enc_in, k_w, None, &[enc_seq, DEC_DIM]);
        let v = b.add_linear(enc_in, v_w, None, &[enc_seq, DEC_DIM]);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &dec_shape);
        let out = b.add_linear(attn, o_w, None, &dec_shape);
        let def = b.build(out).expect("valid kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            const_tensor(&[enc_seq, DEC_DIM], 0.5),
            w(&[DEC_DIM, DEC_DIM]),
            w(&[DEC_DIM, DEC_DIM]),
            w(&[DEC_DIM, DEC_DIM]),
            w(&[DEC_DIM, DEC_DIM]),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = uniform_bounds(&[DEC_SEQ, DEC_DIM], 1.0);
        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        bounds_min_max(&output)
    };

    let (lo_short, hi_short) = build_for_enc_seq(2);
    let (lo_long, hi_long) = build_for_enc_seq(8);

    let width_short = hi_short - lo_short;
    let width_long = hi_long - lo_long;

    eprintln!(
        "GD ext varying enc seq: short(2) width={width_short:.4}, long(8) width={width_long:.4}"
    );
    assert!(width_short.is_finite() && width_long.is_finite());
    // Both must produce valid finite bounds regardless of encoder length
    assert!(lo_short.is_finite() && hi_short.is_finite());
    assert!(lo_long.is_finite() && hi_long.is_finite());
}

// ===========================================================================
// 25. Verify-and-record: cross-attention isolation (IBP)
// ===========================================================================

#[test]
fn test_gd_ext_verify_and_record_cross_attn() {
    let mut b = TensorBlockBuilder::new("gd_ext_vr_xattn");
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, DEC_DIM]);
    let enc_in = b.add_input("enc_features", &[ENC_SEQ, ENC_DIM]);
    let out = add_asymmetric_cross_attn(&mut b, dec_in, enc_in, "xa0");
    let def = b.build(out).expect("valid kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[ENC_SEQ, ENC_DIM], 0.5),
    ];
    bindings.extend(asymmetric_cross_attn_bindings());

    let input = uniform_bounds(&[DEC_SEQ, DEC_DIM], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "granite_docling_enc_dec_ext::test_gd_ext_verify_and_record_cross_attn",
    );
    assert!(result.verification.is_finite);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "GD ext verify-and-record cross-attn: [{lo:.6}, {hi:.6}], mode={:?}",
        result.verification.soundness_mode
    );
}
