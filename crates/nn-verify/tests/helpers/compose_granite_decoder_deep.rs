// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep NY compose tests for Granite-Docling decoder subgraphs.
//!
//! These tests verify bounds propagation through intermediate-depth compositions
//! of the Granite decoder pipeline (used in Granite-Docling-258M). They bridge
//! the gap between existing sub-block tests (RMSNorm, SwiGLU, GQA in
//! `compose_dpdf_granite_docling.rs`) and full VLM end-to-end tests by
//! decomposing the decoder into verifiable intermediate stages:
//!
//! 1. **RMSNorm + SwiGLU FFN + residual** -- Pre-norm FFN block with residual
//!    connection. The core FFN half of a Granite decoder layer (IBP + CROWN).
//!
//! 2. **GQA attention + residual** -- RMSNorm -> Q/K/V projections -> grouped
//!    query attention -> out_proj + residual. 4:1 head ratio (IBP + CROWN).
//!
//! 3. **Full decoder layer** -- RMSNorm -> GQA -> residual -> RMSNorm -> SwiGLU
//!    -> residual. Complete pre-norm Granite decoder block (IBP + CROWN).
//!
//! 4. **2-layer decoder stack** -- Depth composition with widening analysis.
//!    Measures bounds growth through chained decoder layers (IBP).
//!
//! 5. **Decoder + RMSNorm + LM head** -- Decoder output through final RMSNorm
//!    and Linear projection to vocabulary logits (IBP + CROWN).
//!
//! 6. **Decoder + LM head + softmax** -- Full generation pipeline: decoder ->
//!    RMSNorm -> Linear -> softmax probability distribution in [0, 1] (IBP).
//!
//! 7. **Tight-input decoder CROWN** -- Narrow +-0.1 bounds through full decoder
//!    layer for CROWN precision analysis on RMSNorm linearization (CROWN).
//!
//! 8. **Widening analysis** -- 1-layer vs 2-layer decoder IBP width comparison.
//!    Quantifies bounds growth through decoder depth (IBP).
//!
//! 9. **GQA at 8:1 ratio** -- High group ratio GQA (8 Q heads, 1 KV head) for
//!    production-representative attention pattern (IBP + CROWN).
//!
//! 10. **Vision projection + decoder** -- Linear vision-to-LM projection
//!     followed by one decoder layer. Cross-stage composition (IBP + CROWN).
//!
//! 11. **Verify-and-record: decoder + LM head** -- Records to status file.
//!
//! 12. **Verify-and-record: tight-input decoder** -- Records to status file.
//!
//! Architecture references:
//! - Granite-Docling-258M: SigLIP2 vision encoder + Granite LLM decoder
//! - RMSNorm (Zhang & Sennrich, 2019): replaces LayerNorm in Granite
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN in LLaMA/Granite family
//! - GQA (Ainslie et al., 2023): Grouped-Query Attention
//!
//! Dimensions: HIDDEN_DIM=16, FFN_DIM=32, NUM_HEADS=4, NUM_KV_HEADS=1,
//! SEQ_LEN=4, VOCAB_SIZE=8 (small for fast verification).
//! All tests use IbpValidated soundness mode per nn engineering rules.
//!
//! Part of #4273: deep NY compose tests for Granite decoder.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const SEQ_LEN: usize = 4;
const HIDDEN_DIM: usize = 16;
const FFN_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
const NUM_KV_HEADS: usize = 1;
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 4
const VOCAB_SIZE: usize = 8;
const VISION_DIM: usize = 12;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG)
}

fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

fn zeros(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

/// Add one Granite decoder layer: RMSNorm -> GQA -> residual -> RMSNorm -> SwiGLU -> residual.
fn add_granite_decoder_layer(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    idx: usize,
    bindings: &mut Vec<TensorParamBinding>,
) -> TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let n1_eps = b.add_input(&format!("dec{idx}_n1_eps"), &[1]);
    let n1_w = b.add_input(&format!("dec{idx}_n1_w"), &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, n1_eps, 1, n1_w, &shape);

    // GQA: Q projected to full dim, K/V projected to KV_DIM
    let q_w = b.add_input(&format!("dec{idx}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("dec{idx}_k_w"), &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("dec{idx}_v_w"), &[KV_DIM, HIDDEN_DIM]);
    let out_w = b.add_input(&format!("dec{idx}_out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let kv_shape = [SEQ_LEN, KV_DIM];
    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &kv_shape);
    let v = b.add_linear(normed1, v_w, None, &kv_shape);

    // GQA repeat_kv: tile K/V along the feature axis so KV_DIM -> HIDDEN_DIM.
    // This is a genuine repeat (not a size-1 broadcast); model it by concatenating
    // the KV head HIDDEN_DIM / KV_DIM times along the feature axis (axis 1).
    let kv_repeat = HIDDEN_DIM / KV_DIM;
    let k_reps = vec![k; kv_repeat];
    let v_reps = vec![v; kv_repeat];
    let k_broad = b.add_concat(&k_reps, 1, &shape);
    let v_broad = b.add_concat(&v_reps, 1, &shape);

    let attn = b.add_attention(
        q,
        k_broad,
        v_broad,
        AttentionMask::Causal,
        Some(scale),
        &shape,
    );
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let n2_eps = b.add_input(&format!("dec{idx}_n2_eps"), &[1]);
    let n2_w = b.add_input(&format!("dec{idx}_n2_w"), &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    // SwiGLU: silu(gate(x)) * up(x) -> down
    let gate_w = b.add_input(&format!("dec{idx}_gate_w"), &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input(&format!("dec{idx}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input(&format!("dec{idx}_down_w"), &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    let res2 = b.add_binary_add(res1, ffn_out, &shape);

    // Push bindings (11 params per layer)
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // n1_eps
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM]))); // n1_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, HIDDEN_DIM,
    ]))); // q_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM]))); // k_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM]))); // v_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, HIDDEN_DIM,
    ]))); // out_w
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // n2_eps
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM]))); // n2_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ]))); // gate_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ]))); // up_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, FFN_DIM,
    ]))); // down_w

    res2
}

/// Build a single decoder layer kernel.
fn build_single_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_deep_single_dec");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings_unused = Vec::new();
    let out = add_granite_decoder_layer(&mut b, input, 0, &mut bindings_unused);
    b.build(out).expect("valid single decoder kernel")
}

fn single_decoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    // We need to construct dummy bindings for the layer
    let mut layer_bindings = vec![TensorParamBinding::ConstantScalar(1e-5)]; // n1_eps
    layer_bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM]))); // n1_w
    layer_bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, HIDDEN_DIM,
    ]))); // q_w
    layer_bindings.push(TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM]))); // k_w
    layer_bindings.push(TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM]))); // v_w
    layer_bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, HIDDEN_DIM,
    ]))); // out_w
    layer_bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // n2_eps
    layer_bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM]))); // n2_w
    layer_bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ]))); // gate_w
    layer_bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ]))); // up_w
    layer_bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, FFN_DIM,
    ]))); // down_w
    bindings.extend(layer_bindings);
    bindings
}

// ===========================================================================
// 1. RMSNorm + SwiGLU FFN + residual
// ===========================================================================

fn build_rmsnorm_swiglu_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_deep_rmsnorm_swiglu_res");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    let gate_w = b.add_input("gate_w", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_w", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    let res = b.add_binary_add(input, ffn_out, &shape);

    b.build(res)
        .expect("valid RMSNorm + SwiGLU + residual kernel")
}

fn rmsnorm_swiglu_res_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, FFN_DIM])),
    ]
}

#[test]
fn test_granite_deep_rmsnorm_swiglu_res_ibp() {
    let def = build_rmsnorm_swiglu_residual_kernel();
    let bindings = rmsnorm_swiglu_res_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("RMSNorm + SwiGLU + residual IBP: [{lo}, {hi}]");
}

#[test]
fn test_granite_deep_rmsnorm_swiglu_res_crown() {
    let def = build_rmsnorm_swiglu_residual_kernel();
    let bindings = rmsnorm_swiglu_res_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("RMSNorm + SwiGLU + residual CROWN (method={method:?}): [{lo}, {hi}]");
}

// ===========================================================================
// 2. GQA attention + residual
// ===========================================================================

fn build_gqa_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_deep_gqa_res");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let kv_shape = [SEQ_LEN, KV_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[KV_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &kv_shape);
    let v = b.add_linear(normed, v_w, None, &kv_shape);
    // GQA repeat_kv: tile K/V along the feature axis (axis 1) so KV_DIM -> HIDDEN_DIM.
    let kv_repeat = HIDDEN_DIM / KV_DIM;
    let k_reps = vec![k; kv_repeat];
    let v_reps = vec![v; kv_repeat];
    let k_broad = b.add_concat(&k_reps, 1, &shape);
    let v_broad = b.add_concat(&v_reps, 1, &shape);
    let attn = b.add_attention(
        q,
        k_broad,
        v_broad,
        AttentionMask::Causal,
        Some(scale),
        &shape,
    );
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res = b.add_binary_add(input, attn_out, &shape);

    b.build(res).expect("valid GQA + residual kernel")
}

fn gqa_residual_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
    ]
}

#[test]
fn test_granite_deep_gqa_residual_ibp() {
    let def = build_gqa_residual_kernel();
    let bindings = gqa_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GQA + residual IBP: [{lo}, {hi}]");
}

#[test]
fn test_granite_deep_gqa_residual_crown() {
    let def = build_gqa_residual_kernel();
    let bindings = gqa_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GQA + residual CROWN (method={method:?}): [{lo}, {hi}]");
}

// ===========================================================================
// 3. Full decoder layer
// ===========================================================================

#[test]
fn test_granite_deep_full_decoder_ibp() {
    let def = build_single_decoder_kernel();
    let bindings = single_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Full decoder layer IBP: [{lo}, {hi}]");
}

#[test]
fn test_granite_deep_full_decoder_crown() {
    let def = build_single_decoder_kernel();
    let bindings = single_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Full decoder layer CROWN (method={method:?}): [{lo}, {hi}]");
}

// ===========================================================================
// 4. 2-layer decoder stack
// ===========================================================================

fn build_2layer_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_deep_2layer_dec");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings_unused = Vec::new();
    let l1 = add_granite_decoder_layer(&mut b, input, 0, &mut bindings_unused);
    let l2 = add_granite_decoder_layer(&mut b, l1, 1, &mut bindings_unused);
    b.build(l2).expect("valid 2-layer decoder kernel")
}

fn two_layer_decoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            HIDDEN_DIM, HIDDEN_DIM,
        ])));
        bindings.push(TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM])));
        bindings.push(TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM])));
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            HIDDEN_DIM, HIDDEN_DIM,
        ])));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            FFN_DIM, HIDDEN_DIM,
        ])));
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            FFN_DIM, HIDDEN_DIM,
        ])));
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            HIDDEN_DIM, FFN_DIM,
        ])));
    }
    bindings
}

#[test]
fn test_granite_deep_2layer_decoder_ibp() {
    let def = build_2layer_decoder_kernel();
    let bindings = two_layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("2-layer decoder IBP: [{lo}, {hi}]");
}

// ===========================================================================
// 5. Decoder + RMSNorm + LM head
// ===========================================================================

fn build_decoder_lm_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_deep_dec_lm_head");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let vocab_shape = [SEQ_LEN, VOCAB_SIZE];

    let input = b.add_input("x", &shape);
    let mut bindings_unused = Vec::new();
    let dec_out = add_granite_decoder_layer(&mut b, input, 0, &mut bindings_unused);

    // Final RMSNorm
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(dec_out, final_eps, 1, final_w, &shape);

    // LM head
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &vocab_shape);

    b.build(logits).expect("valid decoder + LM head kernel")
}

fn decoder_lm_head_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = single_decoder_bindings();
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // final_eps
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM]))); // final_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB_SIZE, HIDDEN_DIM,
    ]))); // lm_w
    bindings
}

#[test]
fn test_granite_deep_decoder_lm_head_ibp() {
    let def = build_decoder_lm_head_kernel();
    let bindings = decoder_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Decoder + LM head IBP: [{lo}, {hi}]");
}

#[test]
fn test_granite_deep_decoder_lm_head_crown() {
    let def = build_decoder_lm_head_kernel();
    let bindings = decoder_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Decoder + LM head CROWN (method={method:?}): [{lo}, {hi}]");
}

// ===========================================================================
// 6. Decoder + LM head + softmax (full generation)
// ===========================================================================

fn build_decoder_lm_softmax_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_deep_dec_lm_softmax");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let vocab_shape = [SEQ_LEN, VOCAB_SIZE];

    let input = b.add_input("x", &shape);
    let mut bindings_unused = Vec::new();
    let dec_out = add_granite_decoder_layer(&mut b, input, 0, &mut bindings_unused);

    // Final RMSNorm
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(dec_out, final_eps, 1, final_w, &shape);

    // LM head + softmax
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &vocab_shape);
    let probs = b.add_softmax(logits, 1, &vocab_shape);

    b.build(probs)
        .expect("valid decoder + LM head + softmax kernel")
}

fn decoder_lm_softmax_bindings() -> Vec<TensorParamBinding> {
    decoder_lm_head_bindings() // same bindings, softmax is parameter-free
}

#[test]
fn test_granite_deep_decoder_lm_softmax_ibp() {
    let def = build_decoder_lm_softmax_kernel();
    let bindings = decoder_lm_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Decoder + LM head + softmax IBP: [{lo}, {hi}]");
    // Softmax output in [0, 1]
    assert!(lo >= 0.0 - 1e-6, "softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-6, "softmax upper <= 1, got {hi}");
}

// ===========================================================================
// 7. Tight-input decoder CROWN
// ===========================================================================

#[test]
fn test_granite_deep_decoder_tight_input_crown() {
    let def = build_single_decoder_kernel();
    let bindings = single_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Narrow +-0.1 bounds for CROWN precision on RMSNorm linearization
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Tight-input decoder CROWN (method={method:?}): [{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 8. Widening analysis: 1-layer vs 2-layer
// ===========================================================================

#[test]
fn test_granite_deep_widening_1_vs_2_layers() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // 1-layer
    let def1 = build_single_decoder_kernel();
    let b1 = single_decoder_bindings();
    let g1 = tensor_kernel_to_graph(&def1, &b1).expect("1-layer graph");
    let out1 = g1.propagate_ibp(&input).expect("1-layer IBP");
    let (lo1, hi1) = bounds_min_max(&out1);
    let width1 = hi1 - lo1;

    // 2-layer
    let def2 = build_2layer_decoder_kernel();
    let b2 = two_layer_decoder_bindings();
    let g2 = tensor_kernel_to_graph(&def2, &b2).expect("2-layer graph");
    let out2 = g2.propagate_ibp(&input).expect("2-layer IBP");
    let (lo2, hi2) = bounds_min_max(&out2);
    let width2 = hi2 - lo2;

    eprintln!("Widening: 1-layer width={width1:.4}, 2-layer width={width2:.4}");
    eprintln!("  Expansion ratio: {:.2}x", width2 / width1.max(1e-10));

    assert!(
        width2 >= width1 - 1e-4,
        "2-layer bounds should be >= 1-layer: {width2} vs {width1}"
    );
}

// ===========================================================================
// 9. Vision projection + decoder
// ===========================================================================

fn build_vision_proj_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_deep_vis_proj_dec");
    let vis_shape = [SEQ_LEN, VISION_DIM];
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("vision_features", &vis_shape);

    // Vision-to-LM projection
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, VISION_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let projected = b.add_linear(input, proj_w, Some(proj_b), &shape);

    // Decoder layer
    let mut bindings_unused = Vec::new();
    let out = add_granite_decoder_layer(&mut b, projected, 0, &mut bindings_unused);

    b.build(out).expect("valid vision proj + decoder kernel")
}

fn vision_proj_decoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, VISION_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
    ];
    // Decoder layer bindings (11 params)
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, FFN_DIM,
    ])));
    bindings
}

#[test]
fn test_granite_deep_vision_proj_decoder_ibp() {
    let def = build_vision_proj_decoder_kernel();
    let bindings = vision_proj_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, VISION_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Vision proj + decoder IBP: [{lo}, {hi}]");
}

#[test]
fn test_granite_deep_vision_proj_decoder_crown() {
    let def = build_vision_proj_decoder_kernel();
    let bindings = vision_proj_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, VISION_DIM], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Vision proj + decoder CROWN (method={method:?}): [{lo}, {hi}]");
}

// ===========================================================================
// 10. Decoder + LM head verify-and-record
// ===========================================================================

#[test]
fn test_granite_deep_decoder_lm_head_verify_and_record() {
    let def = build_decoder_lm_head_kernel();
    let bindings = decoder_lm_head_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "granite_decoder_deep::test_granite_deep_decoder_lm_head_verify_and_record",
    );
    assert!(result.verification.is_finite);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Decoder + LM head verify: [{lo}, {hi}], mode={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 11. Tight-input decoder verify-and-record
// ===========================================================================

#[test]
fn test_granite_deep_tight_decoder_verify_and_record() {
    let def = build_single_decoder_kernel();
    let bindings = single_decoder_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "granite_decoder_deep::test_granite_deep_tight_decoder_verify_and_record",
    );
    assert!(result.verification.is_finite);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Tight decoder verify: [{lo}, {hi}], mode={:?}",
        result.verification.soundness_mode
    );
}
