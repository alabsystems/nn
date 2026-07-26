// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep NY compose tests for GLM-OCR subgraphs.
//!
//! These tests verify bounds propagation through intermediate-depth compositions
//! of the GLM-OCR decoder pipeline. They bridge the gap between existing
//! sub-block tests (in `compose_dpdf_glm_ocr.rs`) and full end-to-end
//! verification by exercising compositions at increasing depth:
//!
//! 1. **Full decoder layer** — RMSNorm -> GQA attention -> residual ->
//!    RMSNorm -> SwiGLU FFN -> residual. Complete pre-norm decoder block
//!    (IBP + CROWN).
//!
//! 2. **2-layer decoder stack** — Depth composition with widening analysis.
//!    Quantifies bounds growth through chained decoder layers (IBP).
//!
//! 3. **MTP head chain** — Linear -> softmax multi-step prediction heads.
//!    Tests composition of linear projection + softmax normalization
//!    (IBP + CROWN).
//!
//! 4. **Embedding + decoder** — Token embedding -> Linear -> decoder layer.
//!    Cross-stage composition from discrete tokens to continuous hidden
//!    states through a full decoder block (IBP).
//!
//! 5. **Full pipeline** — Embedding -> 2-layer decoder -> LM head -> softmax.
//!    End-to-end bounds from token indices to probability distribution (IBP).
//!
//! 6. **Tight-input analysis** — Narrow +-0.1 bounds for CROWN precision
//!    on the decoder layer. Reduces relaxation gap in RMSNorm and softmax
//!    linearization (IBP + CROWN).
//!
//! Architecture reference:
//! - GLM-4V (THUDM): Vision-language model with GLM-4 decoder for OCR
//! - RMSNorm (Zhang & Sennrich, 2019): replaces LayerNorm in GLM
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN
//! - GQA (Ainslie et al., 2023): Grouped-Query Attention
//!
//! Dimensions are small for fast verification (HIDDEN_DIM=16, SEQ_LEN=4).
//! All tests use IbpValidated soundness mode per nn engineering rules
//! (Sound refuses linearization for normalization layers).
//!
//! Part of #3884: deep NY compose tests for GLM-OCR subgraphs.

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
const HIDDEN_DIM: usize = 16;
/// FFN intermediate dimension (SwiGLU gate and up projections).
const FFN_DIM: usize = 64;
/// Sequence length for decoder sub-block tests.
const SEQ_LEN: usize = 4;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
/// Vocabulary size for embedding/LM head tests.
const VOCAB_SIZE: usize = 32;
/// Weight magnitude for bounded verification.
const W_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
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

/// Add one GLM-OCR decoder layer to the builder.
///
/// Structure: RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
/// Returns the output node ID. Adds 10 input nodes to the builder.
fn add_decoder_layer(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::TensorNodeId,
    layer_idx: usize,
) -> nn_dsl::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let pfx = format!("l{layer_idx}");

    // Pre-attention RMSNorm
    let n1_eps = b.add_input(&format!("{pfx}_n1_eps"), &[1]);
    let n1_w = b.add_input(&format!("{pfx}_n1_w"), &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(x, n1_eps, 1, n1_w, &shape);

    // Self-attention: Q/K/V + causal attention + output projection
    let q_w = b.add_input(&format!("{pfx}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{pfx}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{pfx}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input(&format!("{pfx}_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res1 = b.add_binary_add(x, attn_out, &shape);

    // Pre-FFN RMSNorm
    let n2_eps = b.add_input(&format!("{pfx}_n2_eps"), &[1]);
    let n2_w = b.add_input(&format!("{pfx}_n2_w"), &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    // SwiGLU FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj
    let gate_w = b.add_input(&format!("{pfx}_gate_w"), &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input(&format!("{pfx}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input(&format!("{pfx}_down_w"), &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    b.add_binary_add(res1, ffn_out, &shape)
}

/// Push bindings for one decoder layer (12 parameters).
fn push_decoder_layer_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let norm_w = ones(&[HIDDEN_DIM]);
    let qkvo = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    let gate = w(&[FFN_DIM, HIDDEN_DIM]);
    let up = w(&[FFN_DIM, HIDDEN_DIM]);
    let down = w(&[HIDDEN_DIM, FFN_DIM]);

    // Pre-attention RMSNorm: eps, weight
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
    // Attention: Q, K, V, O
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(qkvo));
    // Pre-FFN RMSNorm: eps, weight
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(norm_w));
    // SwiGLU FFN: gate, up, down
    bindings.push(TensorParamBinding::ConstantTensor(gate));
    bindings.push(TensorParamBinding::ConstantTensor(up));
    bindings.push(TensorParamBinding::ConstantTensor(down));
}

// ===========================================================================
// 1. Full decoder layer: RMSNorm -> GQA attention -> residual ->
//    RMSNorm -> SwiGLU FFN -> residual (IBP + CROWN)
// ===========================================================================

/// Build a complete GLM-OCR decoder layer at small dimensions.
fn build_deep_decoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_deep_decoder_layer");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_decoder_layer(&mut b, input, 0);
    b.build(out).expect("valid deep decoder layer kernel")
}

fn deep_decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_layer_bindings(&mut bindings);
    bindings
}

#[test]
fn test_glm_ocr_deep_decoder_layer_ibp() {
    let def = build_deep_decoder_layer_kernel();
    let bindings = deep_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep decoder layer IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_glm_ocr_deep_decoder_layer_crown() {
    let def = build_deep_decoder_layer_kernel();
    let bindings = deep_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep decoder layer CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_glm_ocr_deep_decoder_layer_verify_and_record() {
    let def = build_deep_decoder_layer_kernel();
    let bindings = deep_decoder_layer_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "glm_ocr_deep_decoder_layer");
    assert_eq!(result.num_variables, 1, "single Variable input");
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM]
    );
}

// ===========================================================================
// 2. 2-layer decoder stack: Depth composition with widening analysis (IBP)
// ===========================================================================

/// Build an N-layer GLM-OCR decoder stack for widening comparison.
fn build_n_layer_decoder(n: usize) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let name = format!("glm_ocr_deep_{n}layer_decoder");
    let mut b = TensorBlockBuilder::new(&name);
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];

    let mut x = input;
    for i in 0..n {
        x = add_decoder_layer(&mut b, x, i);
        push_decoder_layer_bindings(&mut bindings);
    }

    let def = b.build(x).expect("valid n-layer decoder");
    (def, bindings)
}

#[test]
fn test_glm_ocr_deep_2layer_decoder_ibp() {
    let (def, bindings) = build_n_layer_decoder(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep 2-layer decoder IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_glm_ocr_deep_widening_analysis() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // 1-layer
    let (def1, bindings1) = build_n_layer_decoder(1);
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph");
    let output1 = graph1.propagate_ibp(&input).expect("IBP 1-layer");
    let (lo1, hi1) = bounds_min_max(&output1);
    let width1 = hi1 - lo1;

    // 2-layer
    let (def2, bindings2) = build_n_layer_decoder(2);
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("graph");
    let output2 = graph2.propagate_ibp(&input).expect("IBP 2-layer");
    let (lo2, hi2) = bounds_min_max(&output2);
    let width2 = hi2 - lo2;

    eprintln!("GLM-OCR deep decoder widening:");
    eprintln!("  1-layer: [{lo1:.4}, {hi1:.4}], width={width1:.4}");
    eprintln!("  2-layer: [{lo2:.4}, {hi2:.4}], width={width2:.4}");

    assert!(width1.is_finite(), "1-layer width not finite");
    assert!(width2.is_finite(), "2-layer width not finite");
}

// ===========================================================================
// 3. MTP head chain: Linear -> softmax (multi-step prediction heads)
//    (IBP + CROWN)
// ===========================================================================

/// Build a 2-step MTP head chain.
///
/// Step 1: Linear(HIDDEN_DIM -> VOCAB_SIZE) -> softmax
/// Step 2: Linear(VOCAB_SIZE -> HIDDEN_DIM) -> Linear(HIDDEN_DIM -> VOCAB_SIZE) -> softmax
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution).
fn build_deep_mtp_chain_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_deep_mtp_chain");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w1 = b.add_input("lm_w1", &[VOCAB_SIZE, HIDDEN_DIM]);
    let down_w = b.add_input("down_w", &[HIDDEN_DIM, VOCAB_SIZE]);
    let lm_w2 = b.add_input("lm_w2", &[VOCAB_SIZE, HIDDEN_DIM]);

    // Step 1: project to vocab, softmax
    let logits1 = b.add_linear(input, lm_w1, None, &[SEQ_LEN, VOCAB_SIZE]);
    let _probs1 = b.add_softmax(logits1, 1, &[SEQ_LEN, VOCAB_SIZE]);

    // Step 2: project logits back to hidden, then to vocab again
    let hidden2 = b.add_linear(logits1, down_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let logits2 = b.add_linear(hidden2, lm_w2, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs2 = b.add_softmax(logits2, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs2).expect("valid deep MTP chain kernel")
}

fn deep_mtp_chain_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[VOCAB_SIZE, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, VOCAB_SIZE])),
        TensorParamBinding::ConstantTensor(w(&[VOCAB_SIZE, HIDDEN_DIM])),
    ]
}

#[test]
fn test_glm_ocr_deep_mtp_chain_ibp() {
    let def = build_deep_mtp_chain_kernel();
    let bindings = deep_mtp_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep MTP chain IBP: [{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    let eps = 1e-4;
    assert!(lo_min >= -eps, "softmax lower should be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper should be <= 1, got {hi_max}"
    );
}

#[test]
fn test_glm_ocr_deep_mtp_chain_crown() {
    let def = build_deep_mtp_chain_kernel();
    let bindings = deep_mtp_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep MTP chain CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }

    // Softmax invariant holds regardless of propagation method
    let eps = 1e-4;
    assert!(lo_min >= -eps, "softmax lower should be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 4. Embedding + decoder: Token embedding -> Linear -> decoder layer (IBP)
// ===========================================================================

/// Build embedding -> linear projection -> decoder layer.
///
/// Input: `[SEQ_LEN]` (Variable, token indices).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Tests cross-stage composition from discrete tokens through embedding
/// lookup and projection into a full decoder layer.
fn build_deep_embedding_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_deep_embedding_decoder");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("token_ids", &[SEQ_LEN]);
    let emb_w = b.add_input("emb_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Embedding lookup: [SEQ_LEN] -> [SEQ_LEN, HIDDEN_DIM]
    let embedded = b.add_embedding(input, emb_w, &shape);

    // Linear projection
    let projected = b.add_linear(embedded, proj_w, None, &shape);

    // One decoder layer
    let out = add_decoder_layer(&mut b, projected, 0);

    b.build(out).expect("valid deep embedding+decoder kernel")
}

fn deep_embedding_decoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[VOCAB_SIZE, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
    ];
    push_decoder_layer_bindings(&mut bindings);
    bindings
}

#[test]
fn test_glm_ocr_deep_embedding_decoder_ibp() {
    let def = build_deep_embedding_decoder_kernel();
    let bindings = deep_embedding_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Token indices as bounded variable input: indices in [0, VOCAB_SIZE-1]
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), (VOCAB_SIZE - 1) as f32),
    )
    .expect("valid index bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR deep embedding+decoder IBP (indices [0,{}]): [{lo_min}, {hi_max}]",
        VOCAB_SIZE - 1
    );
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. Full pipeline: Embedding -> 2-layer decoder -> LM head -> softmax (IBP)
// ===========================================================================

/// Build the full GLM-OCR pipeline: embedding -> 2-layer decoder -> RMSNorm
/// -> LM head -> softmax.
///
/// Input: `[SEQ_LEN]` (Variable, token indices).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution).
fn build_deep_full_pipeline_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("glm_ocr_deep_full_pipeline");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("token_ids", &[SEQ_LEN]);
    let emb_w = b.add_input("emb_w", &[VOCAB_SIZE, HIDDEN_DIM]);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[VOCAB_SIZE, HIDDEN_DIM])),
    ];

    // Embedding lookup
    let embedded = b.add_embedding(input, emb_w, &shape);

    // 2 decoder layers
    let mut x = embedded;
    for i in 0..2 {
        x = add_decoder_layer(&mut b, x, i);
        push_decoder_layer_bindings(&mut bindings);
    }

    // Final RMSNorm
    let fn_eps = b.add_input("final_n_eps", &[1]);
    let fn_w = b.add_input("final_n_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(x, fn_eps, 1, fn_w, &shape);
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));

    // LM head: Linear -> softmax
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB_SIZE, HIDDEN_DIM,
    ])));

    let def = b.build(probs).expect("valid deep full pipeline kernel");
    (def, bindings)
}

#[test]
fn test_glm_ocr_deep_full_pipeline_ibp() {
    let (def, bindings) = build_deep_full_pipeline_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), (VOCAB_SIZE - 1) as f32),
    )
    .expect("valid index bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR deep full pipeline IBP: [{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    let eps = 1e-4;
    assert!(lo_min >= -eps, "softmax lower should be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper should be <= 1, got {hi_max}"
    );
}

#[test]
fn test_glm_ocr_deep_full_pipeline_verify_and_record() {
    let (def, bindings) = build_deep_full_pipeline_kernel();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), (VOCAB_SIZE - 1) as f32),
    )
    .expect("valid index bounds");

    let result = verify_and_assert(&def, &bindings, &input, "glm_ocr_deep_full_pipeline");
    assert_eq!(result.num_variables, 1, "single Variable input");
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE]
    );
}

// ===========================================================================
// 6. Tight-input analysis: Narrow +-0.1 bounds for CROWN precision
//    (IBP + CROWN)
// ===========================================================================

/// Build a decoder layer tested with narrow input bounds (+-0.1).
///
/// Narrow bounds reduce the relaxation gap in RMSNorm divisor and softmax
/// linearization, allowing CROWN to produce meaningfully tighter results
/// than wide-input IBP.
fn build_deep_tight_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ocr_deep_tight_decoder");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_decoder_layer(&mut b, input, 0);
    b.build(out).expect("valid tight decoder kernel")
}

fn deep_tight_decoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_layer_bindings(&mut bindings);
    bindings
}

#[test]
fn test_glm_ocr_deep_tight_decoder_ibp() {
    let def = build_deep_tight_decoder_kernel();
    let bindings = deep_tight_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Narrow input: +-0.1
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("GLM-OCR deep tight decoder IBP (+-0.1): [{lo_min}, {hi_max}], width={width:.6}");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_glm_ocr_deep_tight_decoder_crown() {
    let def = build_deep_tight_decoder_kernel();
    let bindings = deep_tight_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Narrow input: +-0.1
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!(
        "GLM-OCR deep tight decoder CROWN ({method:?}): [{lo_min}, {hi_max}], width={width:.6}"
    );
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_glm_ocr_deep_tight_decoder_verify_and_record() {
    let def = build_deep_tight_decoder_kernel();
    let bindings = deep_tight_decoder_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);

    let result = verify_and_assert(&def, &bindings, &input, "glm_ocr_deep_tight_decoder");
    assert_eq!(result.num_variables, 1, "single Variable input");
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM]
    );
}
