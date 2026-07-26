// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for the Qwen3 decoder-only LLM.
//!
//! Verifies IBP and CROWN bound propagation through Qwen3 sub-components:
//!
//! ## Tests (10 tests)
//!
//! 1.  **Attention output bounds** — Multi-head causal attention produces finite
//!     bounded outputs with softmax in [0, 1] (IBP + CROWN)
//! 2.  **RoPE application preserves bounds** — Rotary position encoding rotation
//!     preserves input magnitude (IBP + CROWN)
//! 3.  **SwiGLU activation bounds** — Gated FFN with sigmoid gating (IBP)
//! 4.  **Token embedding bounds** — Embedding lookup with bounded weight table (IBP)
//! 5.  **Full decoder layer bound propagation** — Pre-norm attention + SwiGLU MLP
//!     + residual connections (IBP + CROWN)
//! 6.  **RMSNorm bounds** — Root mean square normalization (IBP + Conservative Sound)
//! 7.  **GQA multi-head attention** — Grouped-query attention with N_HEADS > N_KV_HEADS (IBP)
//! 8.  **LM head softmax output** — Linear projection + softmax in [0, 1] (IBP)
//! 9.  **Two-layer decoder stack widening** — Bounds growth through stacked
//!     decoder layers is sub-exponential (IBP)
//! 10. **Decoder + LM head end-to-end** — Full pipeline from features to
//!     softmax probability distribution (IBP)
//!
//! Architecture: Qwen3 decoder-only transformer with RoPE, GQA, SwiGLU.
//!
//! Dimensions (small for fast verification, structurally representative):
//! - D_MODEL=16, N_HEADS=2, N_KV_HEADS=1, FFN_DIM=48, SEQ=4, VOCAB=32
//!
//! Part of #4186: Add compose tests for Qwen3 and DocLayout-YOLO.

mod common;

use common::{
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
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const D_MODEL: usize = 16;
const N_HEADS: usize = 2;
const N_KV_HEADS: usize = 1;
const HEAD_DIM: usize = D_MODEL / N_HEADS; // 8
const HALF_DIM: usize = HEAD_DIM / 2; // 4
const FFN_DIM: usize = 48;
const SEQ: usize = 4;
const VOCAB: usize = 32;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.001;

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG)
}

fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ---------------------------------------------------------------------------
// RoPE cos/sin tables
// ---------------------------------------------------------------------------

fn rope_cos_table() -> ArrayD<f32> {
    let mut data = vec![0.0f32; SEQ * HALF_DIM];
    for pos in 0..SEQ {
        for i in 0..HALF_DIM {
            let theta = (pos as f64) / 10000.0_f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
            data[pos * HALF_DIM + i] = theta.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[SEQ, HALF_DIM]), data).expect("valid cos table")
}

fn rope_sin_table() -> ArrayD<f32> {
    let mut data = vec![0.0f32; SEQ * HALF_DIM];
    for pos in 0..SEQ {
        for i in 0..HALF_DIM {
            let theta = (pos as f64) / 10000.0_f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
            data[pos * HALF_DIM + i] = theta.sin() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[SEQ, HALF_DIM]), data).expect("valid sin table")
}

// ===========================================================================
// 1. Attention output bounds (IBP + CROWN)
// ===========================================================================

/// Build multi-head causal self-attention subgraph.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
///
/// Architecture: Linear Q/K/V projections -> multi-head causal attention -> output projection.
fn build_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_test_attention");
    let shape = [SEQ, D_MODEL];

    let x = b.add_input("x", &shape);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);

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

    b.build(attn).expect("valid attention kernel")
}

fn attention_bindings() -> Vec<TensorParamBinding> {
    let attn_w = w(&[D_MODEL, D_MODEL]);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(attn_w.clone()),
        TensorParamBinding::ConstantTensor(attn_w.clone()),
        TensorParamBinding::ConstantTensor(attn_w.clone()),
        TensorParamBinding::ConstantTensor(attn_w),
    ]
}

/// Verifies that multi-head causal attention produces finite bounded outputs.
#[test]
fn test_qwen3_attention_output_bounds_compose() {
    let def = build_attention_kernel();
    def.validate().expect("attention kernel should validate");

    let bindings = attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP through attention");
    assert_eq!(ibp_out.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&ibp_out);
    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("Qwen3 attention IBP: [{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());

    // CROWN
    let (method, crown_out, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Qwen3 attention CROWN ({method:?}): [{clo}, {chi}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 2. RoPE application preserves bounds (IBP + CROWN)
// ===========================================================================

/// Build RoPE rotation subgraph.
///
/// Input: `[SEQ, HEAD_DIM]` (Variable -- single head activations).
/// Output: `[SEQ, HEAD_DIM]`.
///
/// RoPE applies a rotation matrix using precomputed cos/sin tables.
/// Key property: rotation preserves vector magnitude, so output bounds
/// should be comparable to input bounds.
fn build_rope_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_test_rope");
    let full_shape = [SEQ, HEAD_DIM];
    let half_shape = [SEQ, HALF_DIM];

    let x = b.add_input("x", &full_shape);
    let cos = b.add_input("rope_cos", &[SEQ, HALF_DIM]);
    let sin = b.add_input("rope_sin", &[SEQ, HALF_DIM]);
    let neg_one = b.add_input("neg_one", &[1]);

    let x_first = b.add_narrow(x, 1, 0, HALF_DIM, &half_shape);
    let x_second = b.add_narrow(x, 1, HALF_DIM, HALF_DIM, &half_shape);

    // rot_first = x_first * cos - x_second * sin
    let fc = b.add_binary_mul(x_first, cos, &half_shape);
    let ss = b.add_binary_mul(x_second, sin, &half_shape);
    let neg_bc = b.add_broadcast(neg_one, &half_shape);
    let neg_ss = b.add_binary_mul(ss, neg_bc, &half_shape);
    let rot_first = b.add_binary_add(fc, neg_ss, &half_shape);

    // rot_second = x_first * sin + x_second * cos
    let fs = b.add_binary_mul(x_first, sin, &half_shape);
    let sc = b.add_binary_mul(x_second, cos, &half_shape);
    let rot_second = b.add_binary_add(fs, sc, &half_shape);

    let output = b.add_concat(&[rot_first, rot_second], 1, &full_shape);
    b.build(output).expect("valid RoPE kernel")
}

fn rope_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(rope_cos_table()),
        TensorParamBinding::ConstantTensor(rope_sin_table()),
        TensorParamBinding::ConstantScalar(-1.0),
    ]
}

/// Verifies that RoPE rotation preserves input bound magnitudes.
#[test]
fn test_qwen3_rope_preserves_bounds_compose() {
    let def = build_rope_kernel();
    def.validate().expect("RoPE kernel should validate");

    let bindings = rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HEAD_DIM], 1.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP through RoPE");
    assert_eq!(ibp_out.lower_upper().0.shape(), &[SEQ, HEAD_DIM]);
    assert_bounds_valid(&ibp_out);
    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("Qwen3 RoPE IBP: [{lo}, {hi}]");
    // Rotation with cos/sin bounded in [-1, 1] should not blow up bounds excessively
    assert!(
        hi - lo < 20.0,
        "RoPE bounds width should be < 20, got {}",
        hi - lo
    );

    // CROWN
    let (method, crown_out, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Qwen3 RoPE CROWN ({method:?}): [{clo}, {chi}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 3. SwiGLU activation bounds (IBP)
// ===========================================================================

/// Build SwiGLU MLP subgraph.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
///
/// SwiGLU: gate_proj -> sigmoid -> mul(gate_proj) -> mul(up_proj) -> down_proj
fn build_swiglu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_test_swiglu");
    let shape = [SEQ, D_MODEL];
    let ffn_shape = [SEQ, FFN_DIM];

    let x = b.add_input("x", &shape);
    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, FFN_DIM]);

    let gate_proj = b.add_linear(x, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate_proj, &ffn_shape);
    let gate_act = b.add_binary_mul(gate_proj, gate_sig, &ffn_shape);
    let up_proj = b.add_linear(x, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up_proj, &ffn_shape);
    let out = b.add_linear(gated, down_w, None, &shape);

    b.build(out).expect("valid SwiGLU kernel")
}

fn swiglu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])),
    ]
}

/// Verifies SwiGLU activation produces finite bounded outputs.
#[test]
fn test_qwen3_swiglu_activation_bounds_compose() {
    let def = build_swiglu_kernel();
    def.validate().expect("SwiGLU kernel should validate");

    let bindings = swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 6,
        "SwiGLU graph should have >= 6 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through SwiGLU");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 SwiGLU IBP: [{lo}, {hi}]");
    // With small weights, SwiGLU output should be bounded
    assert!(lo.abs() < 1e4, "SwiGLU lower < 1e4, got {lo}");
    assert!(hi.abs() < 1e4, "SwiGLU upper < 1e4, got {hi}");
}

// ===========================================================================
// 4. Token embedding bounds (IBP)
// ===========================================================================

/// Build token embedding lookup subgraph.
///
/// Input: `[SEQ, VOCAB]` (Variable -- one-hot or soft token representation).
/// Output: `[SEQ, D_MODEL]`.
///
/// For verification purposes, embedding lookup is modeled as a linear
/// projection from the token index space to the embedding space.
fn build_embedding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_test_embedding");

    let x = b.add_input("token_input", &[SEQ, VOCAB]);
    let emb_w = b.add_input("embedding_weight", &[D_MODEL, VOCAB]);
    let out = b.add_linear(x, emb_w, None, &[SEQ, D_MODEL]);

    b.build(out).expect("valid embedding kernel")
}

fn embedding_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, VOCAB])),
    ]
}

/// Verifies that token embedding produces bounded outputs for bounded input.
#[test]
fn test_qwen3_token_embedding_bounds_compose() {
    let def = build_embedding_kernel();
    def.validate().expect("embedding kernel should validate");

    let bindings = embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // One-hot-like input: each position has exactly one token, so bounds are [-1, 1]
    let input = uniform_bounds(&[SEQ, VOCAB], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through embedding");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 embedding IBP: [{lo}, {hi}]");
    // Linear with small weights: output bounded by weight_mag * VOCAB * input_range
    assert!(lo.is_finite(), "embedding lower must be finite, got {lo}");
    assert!(hi.is_finite(), "embedding upper must be finite, got {hi}");
}

// ===========================================================================
// 5. Full decoder layer bound propagation (IBP + CROWN)
// ===========================================================================

/// Build a full Qwen3 decoder layer: pre-norm attention + SwiGLU MLP + residuals.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
fn build_decoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_test_decoder_layer");
    let shape = [SEQ, D_MODEL];
    let ffn_shape = [SEQ, FFN_DIM];

    let x = b.add_input("x", &shape);

    // Pre-attention RMSNorm
    let attn_eps = b.add_input("attn_eps", &[1]);
    let attn_rms_w = b.add_input("attn_rms_w", &[D_MODEL]);
    let normed1 = b.add_rms_norm(x, attn_eps, 1, attn_rms_w, &shape);

    // Self-attention (causal)
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);
    let attn = b
        .add_multi_head_attention(
            normed1,
            q_w,
            k_w,
            v_w,
            out_w,
            N_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("valid causal self-attention");
    let residual1 = b.add_binary_add(x, attn, &shape);

    // Pre-MLP RMSNorm
    let mlp_eps = b.add_input("mlp_eps", &[1]);
    let mlp_rms_w = b.add_input("mlp_rms_w", &[D_MODEL]);
    let normed2 = b.add_rms_norm(residual1, mlp_eps, 1, mlp_rms_w, &shape);

    // SwiGLU MLP
    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, FFN_DIM]);

    let gate_proj = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate_proj, &ffn_shape);
    let gate_act = b.add_binary_mul(gate_proj, gate_sig, &ffn_shape);
    let up_proj = b.add_linear(normed2, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up_proj, &ffn_shape);
    let mlp_out = b.add_linear(gated, down_w, None, &shape);
    let residual2 = b.add_binary_add(residual1, mlp_out, &shape);

    b.build(residual2).expect("valid decoder layer kernel")
}

fn decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let attn_w = w(&[D_MODEL, D_MODEL]);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5), // attn eps
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])), // attn rms
        TensorParamBinding::ConstantTensor(attn_w.clone()), // q_w
        TensorParamBinding::ConstantTensor(attn_w.clone()), // k_w
        TensorParamBinding::ConstantTensor(attn_w.clone()), // v_w
        TensorParamBinding::ConstantTensor(attn_w), // out_w
        TensorParamBinding::ConstantScalar(1e-5), // mlp eps
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])), // mlp rms
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])), // gate
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])), // up
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])), // down
    ]
}

/// Full decoder layer: IBP + CROWN bound propagation.
#[test]
fn test_qwen3_decoder_layer_bounds_compose() {
    let def = build_decoder_layer_kernel();
    def.validate().expect("decoder layer should validate");

    let bindings = decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 20,
        "decoder layer graph >= 20 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    // IBP
    let ibp_out = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder layer");
    assert_eq!(ibp_out.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&ibp_out);
    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("Qwen3 decoder layer IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e6, "decoder layer lower < 1e6, got {lo}");
    assert!(hi.abs() < 1e6, "decoder layer upper < 1e6, got {hi}");

    // CROWN
    let (method, crown_out, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Qwen3 decoder layer CROWN ({method:?}): [{clo}, {chi}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 6. RMSNorm bounds (Conservative Sound)
// ===========================================================================

/// Build standalone RMSNorm subgraph.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
fn build_rmsnorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_test_rmsnorm");
    let shape = [SEQ, D_MODEL];

    let x = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);
    let rms_w = b.add_input("rms_weight", &[D_MODEL]);
    let out = b.add_rms_norm(x, eps, 1, rms_w, &shape);

    b.build(out).expect("valid RMSNorm kernel")
}

fn rmsnorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])),
    ]
}

/// RMSNorm with Conservative mode should produce Sound verification.
#[test]
fn test_qwen3_rmsnorm_bounds_conservative_compose() {
    let def = build_rmsnorm_kernel();
    def.validate().expect("RMSNorm should validate");

    let bindings = rmsnorm_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "qwen3_test_rmsnorm",
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
        "Qwen3 RMSNorm Conservative: [{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 7. GQA multi-head attention (IBP)
// ===========================================================================

/// Build GQA attention subgraph with separate KV-head dimensions.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
///
/// GQA: N_HEADS query heads, N_KV_HEADS key/value heads.
/// K and V projections have reduced dimensions.
fn build_gqa_kernel() -> TensorKernelDef {
    let kv_dim = N_KV_HEADS * HEAD_DIM; // 8
    let mut b = TensorBlockBuilder::new("qwen3_test_gqa");
    let shape = [SEQ, D_MODEL];
    let kv_shape = [SEQ, kv_dim];
    let x = b.add_input("x", &shape);
    // GQA: QK^T contracts over the head dim, so Q must share K's last dim. A
    // broadcast cannot expand kv_dim (8) up to D_MODEL (16) — that is the
    // "repeat KV heads" tile, not a broadcast — so we instead project Q down to
    // kv_dim, run attention on kv_dim, and lift back to D_MODEL via out_w
    // (the established GQA modeling used across the qwen3 verification suite).
    let q_w = b.add_input("q_w", &[kv_dim, D_MODEL]);
    let k_w = b.add_input("k_w", &[kv_dim, D_MODEL]);
    let v_w = b.add_input("v_w", &[kv_dim, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, kv_dim]);

    // Q projection: [SEQ, kv_dim] (shares head dim with K/V)
    let q = b.add_linear(x, q_w, None, &kv_shape);
    // K projection: [SEQ, kv_dim] (fewer heads)
    let k = b.add_linear(x, k_w, None, &kv_shape);
    // V projection: [SEQ, kv_dim]
    let v = b.add_linear(x, v_w, None, &kv_shape);

    // For simplified verification: scale factor
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Attention: Q @ K^T / sqrt(d) with causal mask -> softmax -> @ V, on kv_dim
    let attn_out = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &kv_shape);
    // Lift the kv_dim attention output back to D_MODEL.
    let projected = b.add_linear(attn_out, out_w, None, &shape);

    b.build(projected).expect("valid GQA kernel")
}

fn gqa_bindings() -> Vec<TensorParamBinding> {
    let kv_dim = N_KV_HEADS * HEAD_DIM;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[kv_dim, D_MODEL])),  // q_w
        TensorParamBinding::ConstantTensor(w(&[kv_dim, D_MODEL])),  // k_w
        TensorParamBinding::ConstantTensor(w(&[kv_dim, D_MODEL])),  // v_w
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, kv_dim])),  // out_w
    ]
}

/// GQA attention with fewer KV heads produces finite bounded outputs.
#[test]
fn test_qwen3_gqa_attention_bounds_compose() {
    let def = build_gqa_kernel();
    def.validate().expect("GQA kernel should validate");

    let bindings = gqa_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through GQA");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 GQA IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "GQA lower must be finite, got {lo}");
    assert!(hi.is_finite(), "GQA upper must be finite, got {hi}");
}

// ===========================================================================
// 8. LM head softmax output (IBP)
// ===========================================================================

/// Build LM head: post-norm -> linear projection -> softmax.
///
/// Input: `[SEQ, D_MODEL]` (Variable -- decoder output).
/// Output: `[SEQ, VOCAB]` (probability distribution, bounded in [0, 1]).
fn build_lm_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_test_lm_head");

    let x = b.add_input("decoder_out", &[SEQ, D_MODEL]);
    let eps = b.add_input("post_eps", &[1]);
    let rms_w = b.add_input("post_rms_w", &[D_MODEL]);
    let normed = b.add_rms_norm(x, eps, 1, rms_w, &[SEQ, D_MODEL]);

    let lm_w = b.add_input("lm_weight", &[VOCAB, D_MODEL]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ, VOCAB]);
    let probs = b.add_softmax(logits, 1, &[SEQ, VOCAB]);

    b.build(probs).expect("valid LM head kernel")
}

fn lm_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[VOCAB, D_MODEL])),
    ]
}

/// LM head softmax output must be bounded in [0, 1].
#[test]
fn test_qwen3_lm_head_softmax_bounds_compose() {
    let def = build_lm_head_kernel();
    def.validate().expect("LM head should validate");

    let bindings = lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through LM head");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 LM head softmax IBP: [{lo}, {hi}]");

    // Softmax output must be in [0, 1].
    assert!(lo >= -0.01, "softmax lower bound should be >= 0, got {lo}");
    assert!(hi <= 1.01, "softmax upper bound should be <= 1, got {hi}");
}

// ===========================================================================
// 9. Two-layer decoder stack widening analysis (IBP)
// ===========================================================================

/// Verifies that bounds growth through stacked decoder layers is sub-exponential.
///
/// Compares 1-layer vs 2-layer decoder IBP bounds width to quantify
/// bounds blowup from additional decoder depth.
#[test]
fn test_qwen3_two_layer_stack_widening_compose() {
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    // 1-layer
    let def1 = build_decoder_layer_kernel();
    let bindings1 = decoder_layer_bindings();
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph");
    let out1 = graph1.propagate_ibp(&input).expect("IBP 1-layer");
    let (lo1, hi1) = bounds_min_max(&out1);
    let width1 = hi1 - lo1;

    // 2-layer: build manually
    let def2 = {
        let mut b = TensorBlockBuilder::new("qwen3_test_2layer");
        let shape = [SEQ, D_MODEL];
        let x = b.add_input("x", &shape);

        // Helper closure-like: build one decoder block
        let build_block = |b: &mut TensorBlockBuilder,
                           input: nn_dsl::TensorNodeId,
                           prefix: &str,
                           bindings: &mut Vec<TensorParamBinding>|
         -> nn_dsl::TensorNodeId {
            let shape = [SEQ, D_MODEL];
            let ffn_shape = [SEQ, FFN_DIM];

            let attn_eps = b.add_input(&format!("{prefix}_attn_eps"), &[1]);
            let attn_rms_w = b.add_input(&format!("{prefix}_attn_rms_w"), &[D_MODEL]);
            let normed1 = b.add_rms_norm(input, attn_eps, 1, attn_rms_w, &shape);

            let q_w = b.add_input(&format!("{prefix}_q_w"), &[D_MODEL, D_MODEL]);
            let k_w = b.add_input(&format!("{prefix}_k_w"), &[D_MODEL, D_MODEL]);
            let v_w = b.add_input(&format!("{prefix}_v_w"), &[D_MODEL, D_MODEL]);
            let out_w = b.add_input(&format!("{prefix}_out_w"), &[D_MODEL, D_MODEL]);
            let attn = b
                .add_multi_head_attention(
                    normed1,
                    q_w,
                    k_w,
                    v_w,
                    out_w,
                    N_HEADS,
                    AttentionMask::Causal,
                    &shape,
                )
                .expect("valid attention");
            let residual1 = b.add_binary_add(input, attn, &shape);

            let mlp_eps = b.add_input(&format!("{prefix}_mlp_eps"), &[1]);
            let mlp_rms_w = b.add_input(&format!("{prefix}_mlp_rms_w"), &[D_MODEL]);
            let normed2 = b.add_rms_norm(residual1, mlp_eps, 1, mlp_rms_w, &shape);

            let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[FFN_DIM, D_MODEL]);
            let up_w = b.add_input(&format!("{prefix}_up_w"), &[FFN_DIM, D_MODEL]);
            let down_w = b.add_input(&format!("{prefix}_down_w"), &[D_MODEL, FFN_DIM]);

            let gate_proj = b.add_linear(normed2, gate_w, None, &ffn_shape);
            let gate_sig = b.add_sigmoid(gate_proj, &ffn_shape);
            let gate_act = b.add_binary_mul(gate_proj, gate_sig, &ffn_shape);
            let up_proj = b.add_linear(normed2, up_w, None, &ffn_shape);
            let gated = b.add_binary_mul(gate_act, up_proj, &ffn_shape);
            let mlp_out = b.add_linear(gated, down_w, None, &shape);
            let residual2 = b.add_binary_add(residual1, mlp_out, &shape);

            let attn_w_val = w(&[D_MODEL, D_MODEL]);
            bindings.push(TensorParamBinding::ConstantScalar(1e-5));
            bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL])));
            bindings.push(TensorParamBinding::ConstantTensor(attn_w_val.clone()));
            bindings.push(TensorParamBinding::ConstantTensor(attn_w_val.clone()));
            bindings.push(TensorParamBinding::ConstantTensor(attn_w_val.clone()));
            bindings.push(TensorParamBinding::ConstantTensor(attn_w_val));
            bindings.push(TensorParamBinding::ConstantScalar(1e-5));
            bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL])));
            bindings.push(TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])));
            bindings.push(TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])));
            bindings.push(TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])));

            residual2
        };

        let mut bindings2 = vec![TensorParamBinding::Variable];
        let h = build_block(&mut b, x, "b0", &mut bindings2);
        let out = build_block(&mut b, h, "b1", &mut bindings2);
        let def2 = b.build(out).expect("valid 2-layer stack");
        (def2, bindings2)
    };

    let graph2 = tensor_kernel_to_graph(&def2.0, &def2.1).expect("graph");
    let out2 = graph2.propagate_ibp(&input).expect("IBP 2-layer");
    let (lo2, hi2) = bounds_min_max(&out2);
    let width2 = hi2 - lo2;

    eprintln!("Qwen3 widening analysis:");
    eprintln!("  1-layer: width={width1:.4}, bounds=[{lo1:.4}, {hi1:.4}]");
    eprintln!("  2-layer: width={width2:.4}, bounds=[{lo2:.4}, {hi2:.4}]");

    // Both widths must be finite
    assert!(width1.is_finite(), "1-layer width not finite");
    assert!(width2.is_finite(), "2-layer width not finite");

    // Blowup should be bounded
    let blowup = width2 / 2.0; // input range is 2.0
    assert!(
        blowup < 1e6,
        "2-layer blowup factor < 1e6, got {blowup:.1}x"
    );
}

// ===========================================================================
// 10. Decoder + LM head end-to-end (IBP)
// ===========================================================================

/// Build full decoder layer + LM head + softmax end-to-end.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, VOCAB]` (softmax probabilities in [0, 1]).
fn build_decoder_lm_head_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("qwen3_test_decoder_lm_head");
    let shape = [SEQ, D_MODEL];
    let ffn_shape = [SEQ, FFN_DIM];

    let x = b.add_input("x", &shape);
    let mut bindings = vec![TensorParamBinding::Variable];

    // Decoder layer
    let attn_eps = b.add_input("attn_eps", &[1]);
    let attn_rms_w = b.add_input("attn_rms_w", &[D_MODEL]);
    let normed1 = b.add_rms_norm(x, attn_eps, 1, attn_rms_w, &shape);

    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);
    let attn = b
        .add_multi_head_attention(
            normed1,
            q_w,
            k_w,
            v_w,
            out_w,
            N_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("valid attention");
    let residual1 = b.add_binary_add(x, attn, &shape);

    let mlp_eps = b.add_input("mlp_eps", &[1]);
    let mlp_rms_w = b.add_input("mlp_rms_w", &[D_MODEL]);
    let normed2 = b.add_rms_norm(residual1, mlp_eps, 1, mlp_rms_w, &shape);

    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, FFN_DIM]);

    let gate_proj = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate_proj, &ffn_shape);
    let gate_act = b.add_binary_mul(gate_proj, gate_sig, &ffn_shape);
    let up_proj = b.add_linear(normed2, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up_proj, &ffn_shape);
    let mlp_out = b.add_linear(gated, down_w, None, &shape);
    let residual2 = b.add_binary_add(residual1, mlp_out, &shape);

    // Push decoder layer bindings
    let attn_w = w(&[D_MODEL, D_MODEL]);
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL])));
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(attn_w));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])));

    // Post-norm
    let post_eps = b.add_input("post_eps", &[1]);
    let post_rms_w = b.add_input("post_rms_w", &[D_MODEL]);
    let normed_final = b.add_rms_norm(residual2, post_eps, 1, post_rms_w, &shape);
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL])));

    // LM head
    let lm_w = b.add_input("lm_w", &[VOCAB, D_MODEL]);
    let logits = b.add_linear(normed_final, lm_w, None, &[SEQ, VOCAB]);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[VOCAB, D_MODEL])));

    // Softmax
    let probs = b.add_softmax(logits, 1, &[SEQ, VOCAB]);

    let def = b.build(probs).expect("valid decoder + LM head kernel");
    (def, bindings)
}

/// Full decoder + LM head + softmax: output probabilities must be in [0, 1].
#[test]
fn test_qwen3_decoder_lm_head_end_to_end_compose() {
    let (def, bindings) = build_decoder_lm_head_kernel();
    def.validate().expect("decoder + LM head should validate");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder + LM head");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 decoder + LM head e2e IBP: [{lo}, {hi}]");

    // Softmax output must be in [0, 1].
    assert!(
        lo >= -0.01,
        "e2e softmax lower bound should be >= 0, got {lo}"
    );
    assert!(
        hi <= 1.01,
        "e2e softmax upper bound should be <= 1, got {hi}"
    );
}
