// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep compose tests: FireRed-OCR Qwen3-VL decoder GQA + RoPE composition.
//!
//! Verifies bounds propagation through the Qwen3-VL-2B decoder used in
//! FireRed-OCR for document OCR. These tests target heuristic gaps by
//! testing decoder compositions with RoPE-integrated GQA at depth.
//!
//! 1. **RoPE Q/K application**: cos/sin positional encoding multiplied with
//!    Q/K projections. Bounds remain within [-1, 1] scaled range (IBP + CROWN).
//!
//! 2. **RoPE + GQA attention**: Q with RoPE -> K with RoPE -> softmax ->
//!    V projection. Full attention with positional encoding (IBP + CROWN).
//!
//! 3. **Decoder layer with RoPE**: RMSNorm -> RoPE-GQA -> residual ->
//!    RMSNorm -> SwiGLU FFN -> residual. Complete decoder block (IBP + CROWN).
//!
//! 4. **2-layer decoder stack with RoPE**: Depth composition verifying bounds
//!    widening through chained RoPE-attention + SwiGLU layers (IBP).
//!
//! 5. **Decoder + CTC head**: 2-layer decoder -> RMSNorm -> Linear(HIDDEN, VOCAB)
//!    -> softmax. End-to-end OCR output (IBP + CROWN).
//!
//! 6. **Cross-attention decoder layer**: Encoder features as KV, decoder hidden
//!    as Q, with RMSNorm pre/post. OCR encoder-decoder bridge (IBP + CROWN).
//!
//! 7. **Tight-input RoPE attention**: Narrow +-0.1 bounds for CROWN precision
//!    on RoPE-modulated attention (IBP + CROWN).
//!
//! 8. **Deep encoder-decoder bridge**: Vision encoder output -> projection ->
//!    decoder block -> CTC head. Full encoder-decoder composition (IBP).
//!
//! Architecture reference:
//! - FireRed-OCR: Qwen3-VL-2B variant fine-tuned for document OCR
//! - Qwen3-VL decoder: RMSNorm, GQA, RoPE, SwiGLU
//! - CTC (Graves et al., 2006): Connectionist Temporal Classification
//!
//! Dimensions are small for fast verification (HIDDEN_DIM=16, SEQ_LEN=4).
//! All tests use IbpValidated soundness mode per nn engineering rules.
//!
//! Part of #4304: deep NY compose tests for FireRed-OCR decoder.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 16;
const FFN_DIM: usize = 64;
const SEQ_LEN: usize = 4;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
const VOCAB_SIZE: usize = 64;
const VISION_DIM: usize = 24;
const VISION_SEQ: usize = 4;
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

/// Build cos/sin RoPE tensors (bounded in [-1, 1]).
fn rope_cos_sin() -> (ArrayD<f32>, ArrayD<f32>) {
    let n = SEQ_LEN * HIDDEN_DIM;
    let mut cos_data = vec![0.0f32; n];
    let mut sin_data = vec![0.0f32; n];
    for t in 0..SEQ_LEN {
        for i in 0..HIDDEN_DIM / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / HIDDEN_DIM as f64);
            cos_data[t * HIDDEN_DIM + 2 * i] = freq.cos() as f32;
            cos_data[t * HIDDEN_DIM + 2 * i + 1] = freq.cos() as f32;
            sin_data[t * HIDDEN_DIM + 2 * i] = freq.sin() as f32;
            sin_data[t * HIDDEN_DIM + 2 * i + 1] = freq.sin() as f32;
        }
    }
    (
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), cos_data).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), sin_data).unwrap(),
    )
}

// ===========================================================================
// 1. RoPE Q/K application
// ===========================================================================

fn build_rope_qk_application() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_rope_qk");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let cos = b.add_input("cos", &[SEQ_LEN, HIDDEN_DIM]);
    let sin = b.add_input("sin", &[SEQ_LEN, HIDDEN_DIM]);

    // Q projection
    let q = b.add_linear(input, q_w, None, &shape);
    // RoPE: q * cos + rotate(q) * sin
    // Simplified as element-wise: q * cos (rotation approximation for bounds)
    let q_cos = b.add_binary_mul(q, cos, &shape);
    let q_sin = b.add_binary_mul(q, sin, &shape);
    let q_rope = b.add_binary_add(q_cos, q_sin, &shape);

    b.build(q_rope).expect("valid RoPE Q/K application")
}

fn rope_qk_application_bindings() -> Vec<TensorParamBinding> {
    let (cos, sin) = rope_cos_sin();
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(cos),
        TensorParamBinding::ConstantTensor(sin),
    ]
}

#[test]
fn test_firered_decoder_rope_qk_ibp() {
    let def = build_rope_qk_application();
    let bindings = rope_qk_application_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_firered_decoder_rope_qk_crown() {
    let def = build_rope_qk_application();
    let bindings = rope_qk_application_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered rope_qk CROWN method: {method:?}");
}

// ===========================================================================
// 2. RoPE + GQA attention
// ===========================================================================

fn build_rope_gqa_attention() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_rope_gqa_attn");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let cos = b.add_input("cos", &[SEQ_LEN, HIDDEN_DIM]);
    let sin = b.add_input("sin", &[SEQ_LEN, HIDDEN_DIM]);

    // Q/K projections
    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);

    // Apply RoPE to Q
    let q_cos = b.add_binary_mul(q, cos, &shape);
    let q_sin = b.add_binary_mul(q, sin, &shape);
    let q_rope = b.add_binary_add(q_cos, q_sin, &shape);

    // Apply RoPE to K
    let k_cos = b.add_binary_mul(k, cos, &shape);
    let k_sin = b.add_binary_mul(k, sin, &shape);
    let k_rope = b.add_binary_add(k_cos, k_sin, &shape);

    // Attention with RoPE'd Q/K
    let attn = b.add_attention(
        q_rope,
        k_rope,
        v,
        AttentionMask::Causal,
        Some(scale),
        &shape,
    );
    let out = b.add_linear(attn, o_w, None, &shape);

    b.build(out).expect("valid RoPE + GQA attention")
}

fn rope_gqa_attention_bindings() -> Vec<TensorParamBinding> {
    let (cos, sin) = rope_cos_sin();
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(cos),
        TensorParamBinding::ConstantTensor(sin),
    ]
}

#[test]
fn test_firered_decoder_rope_gqa_attention_ibp() {
    let def = build_rope_gqa_attention();
    let bindings = rope_gqa_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
}

#[test]
fn test_firered_decoder_rope_gqa_attention_crown() {
    let def = build_rope_gqa_attention();
    let bindings = rope_gqa_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered rope_gqa_attention CROWN method: {method:?}");
}

// ===========================================================================
// 3. Decoder layer with RoPE
// ===========================================================================

fn add_firered_decoder_layer(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::TensorNodeId,
    pfx: &str,
) -> nn_dsl::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let n1_eps = b.add_input(&format!("{pfx}_n1_eps"), &[1]);
    let n1_w = b.add_input(&format!("{pfx}_n1_w"), &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(x, n1_eps, 1, n1_w, &shape);

    // Q/K/V projection
    let q_w = b.add_input(&format!("{pfx}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{pfx}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{pfx}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input(&format!("{pfx}_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let cos = b.add_input(&format!("{pfx}_cos"), &[SEQ_LEN, HIDDEN_DIM]);
    let sin = b.add_input(&format!("{pfx}_sin"), &[SEQ_LEN, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);

    // RoPE on Q/K
    let q_cos = b.add_binary_mul(q, cos, &shape);
    let q_sin = b.add_binary_mul(q, sin, &shape);
    let q_rope = b.add_binary_add(q_cos, q_sin, &shape);
    let k_cos = b.add_binary_mul(k, cos, &shape);
    let k_sin = b.add_binary_mul(k, sin, &shape);
    let k_rope = b.add_binary_add(k_cos, k_sin, &shape);

    let attn = b.add_attention(
        q_rope,
        k_rope,
        v,
        AttentionMask::Causal,
        Some(scale),
        &shape,
    );
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res1 = b.add_binary_add(x, attn_out, &shape);

    // Pre-FFN RMSNorm
    let n2_eps = b.add_input(&format!("{pfx}_n2_eps"), &[1]);
    let n2_w = b.add_input(&format!("{pfx}_n2_w"), &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    // SwiGLU FFN
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

fn push_firered_decoder_layer_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let (cos, sin) = rope_cos_sin();
    // RMSNorm 1
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    // Q/K/V/O
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            HIDDEN_DIM, HIDDEN_DIM,
        ])));
    }
    // RoPE cos/sin
    bindings.push(TensorParamBinding::ConstantTensor(cos));
    bindings.push(TensorParamBinding::ConstantTensor(sin));
    // RMSNorm 2
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    // SwiGLU
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

fn build_firered_decoder_layer() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_decoder_layer");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_firered_decoder_layer(&mut b, input, "l0");
    b.build(out).expect("valid FireRed decoder layer")
}

fn firered_decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_firered_decoder_layer_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_decoder_layer_rope_ibp() {
    let def = build_firered_decoder_layer();
    let bindings = firered_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_firered_decoder_layer_rope_crown() {
    let def = build_firered_decoder_layer();
    let bindings = firered_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered decoder_layer CROWN method: {method:?}");
}

// ===========================================================================
// 4. 2-layer decoder stack with RoPE
// ===========================================================================

fn build_firered_two_layer_decoder() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_2layer_decoder");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let x = add_firered_decoder_layer(&mut b, input, "l0");
    let x = add_firered_decoder_layer(&mut b, x, "l1");
    b.build(x).expect("valid 2-layer FireRed decoder")
}

fn firered_two_layer_decoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_firered_decoder_layer_bindings(&mut bindings);
    push_firered_decoder_layer_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_decoder_2layer_stack_ibp() {
    let def = build_firered_two_layer_decoder();
    let bindings = firered_two_layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    // Check that bounds don't blow up vacuously
    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;
    assert!(width < 1e6, "2-layer decoder bounds too wide: {width}");
    eprintln!("2-layer decoder IBP width: {width:.4}");
}

// ===========================================================================
// 5. Decoder + CTC head
// ===========================================================================

fn build_firered_decoder_ctc() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_decoder_ctc");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let x = add_firered_decoder_layer(&mut b, input, "l0");
    let x = add_firered_decoder_layer(&mut b, x, "l1");

    // Final RMSNorm
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(x, final_eps, 1, final_w, &shape);

    // CTC head
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, ctc_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    b.build(probs).expect("valid decoder + CTC head")
}

fn firered_decoder_ctc_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_firered_decoder_layer_bindings(&mut bindings);
    push_firered_decoder_layer_bindings(&mut bindings);
    // Final RMSNorm
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    // CTC head
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB_SIZE, HIDDEN_DIM,
    ])));
    bindings
}

#[test]
fn test_firered_decoder_ctc_ibp() {
    let def = build_firered_decoder_ctc();
    let bindings = firered_decoder_ctc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "CTC softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "CTC softmax upper <= 1, got {hi}");
}

#[test]
fn test_firered_decoder_ctc_crown() {
    let def = build_firered_decoder_ctc();
    let bindings = firered_decoder_ctc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered decoder_ctc CROWN method: {method:?}");
}

// ===========================================================================
// 6. Cross-attention decoder layer
// ===========================================================================

fn build_firered_cross_attention_decoder() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_cross_attn_decoder");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Pre-attention RMSNorm
    let n1_eps = b.add_input("n1_eps", &[1]);
    let n1_w = b.add_input("n1_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, n1_eps, 1, n1_w, &shape);

    // Cross-attention: Q from decoder, K/V reuse from same normed (simplified)
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res = b.add_binary_add(input, attn_out, &shape);

    // Post RMSNorm
    let n2_eps = b.add_input("n2_eps", &[1]);
    let n2_w = b.add_input("n2_w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(res, n2_eps, 1, n2_w, &shape);

    b.build(out).expect("valid cross-attention decoder layer")
}

fn firered_cross_attention_decoder_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
    ]
}

#[test]
fn test_firered_cross_attention_decoder_ibp() {
    let def = build_firered_cross_attention_decoder();
    let bindings = firered_cross_attention_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
}

#[test]
fn test_firered_cross_attention_decoder_crown() {
    let def = build_firered_cross_attention_decoder();
    let bindings = firered_cross_attention_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered cross_attention_decoder CROWN method: {method:?}");
}

// ===========================================================================
// 7. Tight-input RoPE attention
// ===========================================================================

#[test]
fn test_firered_decoder_layer_rope_tight_crown() {
    let def = build_firered_decoder_layer();
    let bindings = firered_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    // Compare with wide bounds
    let wide_ibp = graph
        .propagate_ibp(&uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0))
        .expect("wide IBP");
    let (_, tight_hi) = bounds_min_max(&output);
    let (_, wide_hi) = bounds_min_max(&wide_ibp);
    eprintln!("tight CROWN max: {tight_hi:.4}, wide IBP max: {wide_hi:.4}, method: {method:?}");
}

// ===========================================================================
// 8. Vision encoder output -> projection -> decoder block -> CTC head
// ===========================================================================

fn build_firered_vision_decoder_ctc() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vision_decoder_ctc");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Vision encoder output (already encoded)
    let input = b.add_input("vision_features", &[VISION_SEQ, VISION_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, VISION_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let projected = b.add_linear(input, proj_w, Some(proj_b), &shape);

    // One decoder layer
    let x = add_firered_decoder_layer(&mut b, projected, "l0");

    // CTC head
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(x, final_eps, 1, final_w, &shape);
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, ctc_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    b.build(probs).expect("valid vision -> decoder -> CTC")
}

fn firered_vision_decoder_ctc_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, VISION_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
    ];
    push_firered_decoder_layer_bindings(&mut bindings);
    // Final RMSNorm
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    // CTC head
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB_SIZE, HIDDEN_DIM,
    ])));
    bindings
}

#[test]
fn test_firered_vision_decoder_ctc_ibp() {
    let def = build_firered_vision_decoder_ctc();
    let bindings = firered_vision_decoder_ctc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "softmax upper <= 1, got {hi}");
}

// ===========================================================================
// Verify-and-record
// ===========================================================================

#[test]
fn test_firered_decoder_rope_gqa_verify_and_record() {
    let def = build_rope_gqa_attention();
    let bindings = rope_gqa_attention_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "firered_ocr::test_firered_decoder_rope_gqa_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}
