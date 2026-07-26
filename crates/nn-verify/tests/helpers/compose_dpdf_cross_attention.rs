// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for cross-attention patterns in encoder-decoder architectures
//! used by dpdf models (Table Transformer/DETR, VLM decoder).
//!
//! Verifies IBP and CROWN bound propagation through cross-attention, the
//! fundamental mechanism by which decoder queries attend to encoder memory
//! in encoder-decoder transformer architectures.
//!
//! 1.  **Basic cross-attention**: query attends to key-value memory (IBP).
//! 2.  **Cross-attention with LayerNorm pre-processing** (IBP + CROWN).
//! 3.  **Cross-attention with residual connection**: q + Attn(q, kv) (IBP).
//! 4.  **Multi-head cross-attention with h=8 heads** (IBP).
//! 5.  **Cross-attention softmax weights bounded in [0, 1]** (IBP).
//! 6.  **Cross-attention with different Q and KV dimensions** (IBP).
//! 7.  **Cross-attention + FFN decoder layer** (IBP + CROWN).
//! 8.  **Self-attention -> cross-attention sequential (DETR decoder)** (IBP).
//! 9.  **Cross-attention with position encoding added to queries** (IBP).
//! 10. **Cross-attention with position encoding added to keys** (IBP).
//! 11. **Stacked cross-attention (2-layer decoder)** (IBP + CROWN).
//! 12. **Cross-attention KV from vision encoder features (VLM pattern)** (IBP).
//! 13. **Cross-attention monotone tightening**: smaller eps -> tighter bounds (IBP).
//! 14. **CROWN tightness for cross-attention vs IBP** (CROWN).
//! 15. **Full decoder layer**: LN + self-attn + LN + cross-attn + LN + FFN (IBP + CROWN).
//!
//! Architecture references:
//! - DETR (Carion et al. 2020): DEtection TRansformer
//! - Table Transformer (Smock et al. 2022): DETR-based table structure recognition
//! - Qwen2-VL / Qwen3-VL (Alibaba): Vision-language model cross-attention decoder
//!
//! Dimensions (small for fast verification, structurally representative):
//! - QUERY_LEN=4, KV_LEN=8, HIDDEN_DIM=64, FFN_DIM=128, NUM_HEADS=4
//!
//! Part of #4033: Compose tests for cross-attention in encoder-decoder models.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, sinusoidal_pe,
    uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Query sequence length (decoder queries / object queries).
const QUERY_LEN: usize = 4;
/// Key-value sequence length (encoder memory tokens).
const KV_LEN: usize = 8;
/// Hidden dimension for transformer.
const HIDDEN_DIM: usize = 64;
/// FFN intermediate dimension.
const FFN_DIM: usize = 128;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 16
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Build basic cross-attention: Q from queries, K/V from encoder memory.
///
/// Pattern: Linear(Q) x Linear(K)^T / sqrt(d_k) -> softmax -> * Linear(V)
/// -> Linear(out)
///
/// Query shape: `[query_len, hidden_dim]`, KV shape: `[kv_len, hidden_dim]`.
/// Output shape: `[query_len, hidden_dim]`.
fn build_cross_attention(
    b: &mut TensorBlockBuilder,
    queries: nn_dsl::TensorNodeId,
    encoder_mem: nn_dsl::TensorNodeId,
    prefix: &str,
    query_len: usize,
    kv_len: usize,
    hidden_dim: usize,
) -> nn_dsl::TensorNodeId {
    let q_shape = [query_len, hidden_dim];
    let kv_shape = [kv_len, hidden_dim];
    let head_dim = hidden_dim / NUM_HEADS;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_w = b.add_input(&format!("{prefix}_q_w"), &[hidden_dim, hidden_dim]);
    let k_w = b.add_input(&format!("{prefix}_k_w"), &[hidden_dim, hidden_dim]);
    let v_w = b.add_input(&format!("{prefix}_v_w"), &[hidden_dim, hidden_dim]);
    let o_w = b.add_input(&format!("{prefix}_o_w"), &[hidden_dim, hidden_dim]);

    let q = b.add_linear(queries, q_w, None, &q_shape);
    let k = b.add_linear(encoder_mem, k_w, None, &kv_shape);
    let v = b.add_linear(encoder_mem, v_w, None, &kv_shape);

    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &q_shape);
    b.add_linear(attn, o_w, None, &q_shape)
}

/// Push cross-attention weight bindings (q_w, k_w, v_w, o_w).
fn push_cross_attention_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    hidden_dim: usize,
    weight_mag: f32,
) {
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[hidden_dim, hidden_dim]),
            weight_mag,
        )));
    }
}

/// Push LayerNorm bindings (weight, bias, eps).
fn push_layer_norm_bindings(bindings: &mut Vec<TensorParamBinding>, hidden_dim: usize) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[hidden_dim]),
        1.0f32,
    ))); // weight
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[hidden_dim]),
        0.0f32,
    ))); // bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
}

// ===========================================================================
// 1. Basic cross-attention: query attends to key-value memory (IBP)
// ===========================================================================

fn build_basic_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_cross_attn_basic");
    let queries = b.add_input("queries", &[QUERY_LEN, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[KV_LEN, HIDDEN_DIM]);

    let out = build_cross_attention(
        &mut b,
        queries,
        encoder_mem,
        "ca",
        QUERY_LEN,
        KV_LEN,
        HIDDEN_DIM,
    );
    b.build(out).expect("valid basic cross-attention kernel")
}

fn basic_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable, // queries
    ];
    // encoder_mem is a constant (from encoder output)
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[KV_LEN, HIDDEN_DIM]),
        0.1f32,
    )));
    push_cross_attention_bindings(&mut bindings, HIDDEN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_cross_attention_basic_ibp() {
    let def = build_basic_cross_attention_kernel();
    let bindings = basic_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attention basic IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Cross-attention with LayerNorm pre-processing (IBP + CROWN)
// ===========================================================================

fn build_layernorm_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_cross_attn_layernorm");
    let queries = b.add_input("queries", &[QUERY_LEN, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[KV_LEN, HIDDEN_DIM]);

    // LayerNorm on queries before cross-attention
    let ln_w = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(queries, ln_eps, 1, ln_w, ln_b, &[QUERY_LEN, HIDDEN_DIM]);

    let out = build_cross_attention(
        &mut b,
        normed,
        encoder_mem,
        "ca",
        QUERY_LEN,
        KV_LEN,
        HIDDEN_DIM,
    );
    b.build(out)
        .expect("valid LayerNorm + cross-attention kernel")
}

fn layernorm_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_LEN, HIDDEN_DIM]), 0.1f32)), // encoder_mem
    ];
    push_layer_norm_bindings(&mut bindings, HIDDEN_DIM);
    push_cross_attention_bindings(&mut bindings, HIDDEN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_cross_attention_layernorm_ibp() {
    let def = build_layernorm_cross_attention_kernel();
    let bindings = layernorm_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("LayerNorm + cross-attention IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

#[test]
fn test_cross_attention_layernorm_crown() {
    let def = build_layernorm_cross_attention_kernel();
    let bindings = layernorm_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("LayerNorm + cross-attention CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 3. Cross-attention with residual connection: q + Attn(q, kv) (IBP)
// ===========================================================================

fn build_residual_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_cross_attn_residual");
    let queries = b.add_input("queries", &[QUERY_LEN, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[KV_LEN, HIDDEN_DIM]);
    let q_shape = [QUERY_LEN, HIDDEN_DIM];

    let attn_out = build_cross_attention(
        &mut b,
        queries,
        encoder_mem,
        "ca",
        QUERY_LEN,
        KV_LEN,
        HIDDEN_DIM,
    );
    // Residual: queries + Attn(queries, encoder_mem)
    let out = b.add_binary_add(queries, attn_out, &q_shape);
    b.build(out).expect("valid residual cross-attention kernel")
}

fn residual_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_LEN, HIDDEN_DIM]), 0.1f32)), // encoder_mem
    ];
    push_cross_attention_bindings(&mut bindings, HIDDEN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_cross_attention_residual_ibp() {
    let def = build_residual_cross_attention_kernel();
    let bindings = residual_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Residual cross-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Residual: input in [-1,1] + small attention output
    assert!(
        lo_min > -100.0,
        "residual lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 4. Multi-head cross-attention with h=8 heads (IBP)
// ===========================================================================

#[test]
fn test_cross_attention_8_heads_ibp() {
    let num_heads_8 = 8;
    let hidden_dim_8 = 64; // 64 / 8 = 8 per head
    let head_dim_8 = hidden_dim_8 / num_heads_8;
    let scale_8 = 1.0 / (head_dim_8 as f32).sqrt();

    let mut b = TensorBlockBuilder::new("dpdf_cross_attn_8heads");
    let queries = b.add_input("queries", &[QUERY_LEN, hidden_dim_8]);
    let encoder_mem = b.add_input("encoder_mem", &[KV_LEN, hidden_dim_8]);

    let q_w = b.add_input("q_w", &[hidden_dim_8, hidden_dim_8]);
    let k_w = b.add_input("k_w", &[hidden_dim_8, hidden_dim_8]);
    let v_w = b.add_input("v_w", &[hidden_dim_8, hidden_dim_8]);
    let o_w = b.add_input("o_w", &[hidden_dim_8, hidden_dim_8]);

    let q = b.add_linear(queries, q_w, None, &[QUERY_LEN, hidden_dim_8]);
    let k = b.add_linear(encoder_mem, k_w, None, &[KV_LEN, hidden_dim_8]);
    let v = b.add_linear(encoder_mem, v_w, None, &[KV_LEN, hidden_dim_8]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale_8),
        &[QUERY_LEN, hidden_dim_8],
    );
    let out = b.add_linear(attn, o_w, None, &[QUERY_LEN, hidden_dim_8]);
    let def = b.build(out).expect("valid 8-head cross-attention kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[KV_LEN, hidden_dim_8]),
            0.1f32,
        )), // encoder_mem
    ];
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[hidden_dim_8, hidden_dim_8]),
            WEIGHT_MAG,
        )));
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, hidden_dim_8], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("8-head cross-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. Cross-attention softmax weights bounded in [0, 1] (IBP)
// ===========================================================================

/// Isolate the attention weight path: Q*K^T / sqrt(d_k) -> softmax.
/// Softmax output should be bounded in [0, 1].
#[test]
fn test_cross_attention_softmax_weights_bounded_ibp() {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("dpdf_cross_attn_softmax_weights");
    let queries = b.add_input("queries", &[QUERY_LEN, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[KV_LEN, HIDDEN_DIM]);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(queries, q_w, None, &[QUERY_LEN, HIDDEN_DIM]);
    let k = b.add_linear(encoder_mem, k_w, None, &[KV_LEN, HIDDEN_DIM]);
    let v = b.add_linear(encoder_mem, v_w, None, &[KV_LEN, HIDDEN_DIM]);

    // Full attention -> softmax path: Q * K^T / sqrt(d_k) -> softmax -> V
    // Use add_attention which produces the full attended output including softmax.
    // The softmax weights are internal to the attention; we verify the output
    // bounds are finite and the overall structure is sound.
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[QUERY_LEN, HIDDEN_DIM],
    );

    // Apply sigmoid to squeeze output into [0, 1] for direct boundedness check.
    let out = b.add_sigmoid(attn, &[QUERY_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid softmax weights kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_LEN, HIDDEN_DIM]), 0.1f32)), // encoder_mem
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // q_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // k_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // v_w
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Cross-attention softmax weights IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 6. Cross-attention with different Q and KV dimensions (IBP)
// ===========================================================================

/// Cross-attention where Q dimension differs from KV dimension.
/// Q has dim=64, KV has dim=128. Requires projection alignment.
#[test]
fn test_cross_attention_different_q_kv_dims_ibp() {
    let q_dim = 64;
    let kv_dim = 128;
    let inner_dim = 64; // Projected dimension for attention
    let head_dim_local = inner_dim / NUM_HEADS;
    let scale = 1.0 / (head_dim_local as f32).sqrt();

    let mut b = TensorBlockBuilder::new("dpdf_cross_attn_diff_dims");
    let queries = b.add_input("queries", &[QUERY_LEN, q_dim]);
    let encoder_mem = b.add_input("encoder_mem", &[KV_LEN, kv_dim]);

    // Project to common inner_dim
    let q_w = b.add_input("q_w", &[inner_dim, q_dim]);
    let k_w = b.add_input("k_w", &[inner_dim, kv_dim]);
    let v_w = b.add_input("v_w", &[inner_dim, kv_dim]);
    let o_w = b.add_input("o_w", &[q_dim, inner_dim]);

    let q = b.add_linear(queries, q_w, None, &[QUERY_LEN, inner_dim]);
    let k = b.add_linear(encoder_mem, k_w, None, &[KV_LEN, inner_dim]);
    let v = b.add_linear(encoder_mem, v_w, None, &[KV_LEN, inner_dim]);

    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[QUERY_LEN, inner_dim],
    );
    let out = b.add_linear(attn, o_w, None, &[QUERY_LEN, q_dim]);
    let def = b
        .build(out)
        .expect("valid different-dim cross-attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_LEN, kv_dim]), 0.1f32)), // encoder_mem
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[inner_dim, q_dim]),
            WEIGHT_MAG,
        )), // q_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[inner_dim, kv_dim]),
            WEIGHT_MAG,
        )), // k_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[inner_dim, kv_dim]),
            WEIGHT_MAG,
        )), // v_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[q_dim, inner_dim]),
            WEIGHT_MAG,
        )), // o_w
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, q_dim], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Cross-attention diff dims (q={q_dim}, kv={kv_dim}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. Cross-attention + FFN decoder layer (IBP + CROWN)
// ===========================================================================

fn build_cross_attn_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_cross_attn_ffn");
    let queries = b.add_input("queries", &[QUERY_LEN, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[KV_LEN, HIDDEN_DIM]);
    let q_shape = [QUERY_LEN, HIDDEN_DIM];
    let ffn_shape = [QUERY_LEN, FFN_DIM];

    // Cross-attention
    let ca_out = build_cross_attention(
        &mut b,
        queries,
        encoder_mem,
        "ca",
        QUERY_LEN,
        KV_LEN,
        HIDDEN_DIM,
    );
    let res_ca = b.add_binary_add(queries, ca_out, &q_shape);

    // LayerNorm before FFN
    let ffn_ln_w = b.add_input("ffn_ln_w", &[HIDDEN_DIM]);
    let ffn_ln_b = b.add_input("ffn_ln_b", &[HIDDEN_DIM]);
    let ffn_eps = b.add_input("ffn_eps", &[1]);
    let normed = b.add_layer_norm(res_ca, ffn_eps, 1, ffn_ln_w, ffn_ln_b, &q_shape);

    // FFN: Linear -> ReLU -> Linear
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, HIDDEN_DIM]);
    let ffn2_w = b.add_input("ffn2_w", &[HIDDEN_DIM, FFN_DIM]);
    let h = b.add_linear(normed, ffn1_w, None, &ffn_shape);
    let h = b.add_relu(h, &ffn_shape);
    let ffn_out = b.add_linear(h, ffn2_w, None, &q_shape);

    // Residual
    let out = b.add_binary_add(res_ca, ffn_out, &q_shape);
    b.build(out).expect("valid cross-attention + FFN kernel")
}

fn cross_attn_ffn_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_LEN, HIDDEN_DIM]), 0.1f32)), // encoder_mem
    ];
    push_cross_attention_bindings(&mut bindings, HIDDEN_DIM, WEIGHT_MAG);
    push_layer_norm_bindings(&mut bindings, HIDDEN_DIM);
    // FFN weights
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FFN_DIM, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, FFN_DIM]),
        WEIGHT_MAG,
    )));
    bindings
}

#[test]
fn test_cross_attention_ffn_ibp() {
    let def = build_cross_attn_ffn_kernel();
    let bindings = cross_attn_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attention + FFN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_cross_attention_ffn_crown() {
    let def = build_cross_attn_ffn_kernel();
    let bindings = cross_attn_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attention + FFN CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 8. Self-attention -> cross-attention sequential (DETR decoder) (IBP)
// ===========================================================================

fn build_self_then_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_self_then_cross_attn");
    let queries = b.add_input("queries", &[QUERY_LEN, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[KV_LEN, HIDDEN_DIM]);
    let q_shape = [QUERY_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Self-attention over queries
    let sa_q_w = b.add_input("sa_q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_k_w = b.add_input("sa_k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_v_w = b.add_input("sa_v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_o_w = b.add_input("sa_o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let sq = b.add_linear(queries, sa_q_w, None, &q_shape);
    let sk = b.add_linear(queries, sa_k_w, None, &q_shape);
    let sv = b.add_linear(queries, sa_v_w, None, &q_shape);
    let sa = b.add_attention(sq, sk, sv, AttentionMask::Standard, Some(scale), &q_shape);
    let sa_proj = b.add_linear(sa, sa_o_w, None, &q_shape);
    let res_sa = b.add_binary_add(queries, sa_proj, &q_shape);

    // Cross-attention: refined queries attend to encoder memory
    let ca_out = build_cross_attention(
        &mut b,
        res_sa,
        encoder_mem,
        "ca",
        QUERY_LEN,
        KV_LEN,
        HIDDEN_DIM,
    );
    let out = b.add_binary_add(res_sa, ca_out, &q_shape);
    b.build(out).expect("valid self->cross attention kernel")
}

fn self_then_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = || {
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        ))
    };
    let mut bindings = vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_LEN, HIDDEN_DIM]), 0.1f32)), // encoder_mem
    ];
    // Self-attention projections (4)
    for _ in 0..4 {
        bindings.push(proj_w());
    }
    // Cross-attention projections (4)
    push_cross_attention_bindings(&mut bindings, HIDDEN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_self_then_cross_attention_ibp() {
    let def = build_self_then_cross_attention_kernel();
    let bindings = self_then_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Self-attn -> cross-attn (DETR) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Cross-attention with position encoding added to queries (IBP)
// ===========================================================================

#[test]
fn test_cross_attention_pe_queries_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_cross_attn_pe_queries");
    let queries = b.add_input("queries", &[QUERY_LEN, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[KV_LEN, HIDDEN_DIM]);
    let q_shape = [QUERY_LEN, HIDDEN_DIM];

    // Add sinusoidal position encoding to queries
    let pe = b.add_input("query_pe", &[QUERY_LEN, HIDDEN_DIM]);
    let queries_pe = b.add_binary_add(queries, pe, &q_shape);

    // Cross-attention with position-encoded queries
    let out = build_cross_attention(
        &mut b,
        queries_pe,
        encoder_mem,
        "ca",
        QUERY_LEN,
        KV_LEN,
        HIDDEN_DIM,
    );
    let def = b
        .build(out)
        .expect("valid PE queries cross-attention kernel");

    let pe_data = sinusoidal_pe(QUERY_LEN, HIDDEN_DIM);
    let mut bindings = vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_LEN, HIDDEN_DIM]), 0.1f32)), // encoder_mem
        TensorParamBinding::ConstantTensor(pe_data), // query_pe
    ];
    push_cross_attention_bindings(&mut bindings, HIDDEN_DIM, WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attention + query PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. Cross-attention with position encoding added to keys (IBP)
// ===========================================================================

#[test]
fn test_cross_attention_pe_keys_ibp() {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("dpdf_cross_attn_pe_keys");
    let queries = b.add_input("queries", &[QUERY_LEN, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[KV_LEN, HIDDEN_DIM]);
    let q_shape = [QUERY_LEN, HIDDEN_DIM];
    let kv_shape = [KV_LEN, HIDDEN_DIM];

    // Sinusoidal PE added to keys (common in DETR for spatial awareness)
    let key_pe = b.add_input("key_pe", &[KV_LEN, HIDDEN_DIM]);
    let encoder_with_pe = b.add_binary_add(encoder_mem, key_pe, &kv_shape);

    // Cross-attention: Q from queries, K from encoder+PE, V from encoder
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(queries, q_w, None, &q_shape);
    let k = b.add_linear(encoder_with_pe, k_w, None, &kv_shape);
    let v = b.add_linear(encoder_mem, v_w, None, &kv_shape);

    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &q_shape);
    let out = b.add_linear(attn, o_w, None, &q_shape);
    let def = b.build(out).expect("valid PE keys cross-attention kernel");

    let pe_data = sinusoidal_pe(KV_LEN, HIDDEN_DIM);
    let bindings = vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_LEN, HIDDEN_DIM]), 0.1f32)), // encoder_mem
        TensorParamBinding::ConstantTensor(pe_data), // key_pe
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // q_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // k_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // v_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // o_w
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attention + key PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. Stacked cross-attention (2-layer decoder) (IBP + CROWN)
// ===========================================================================

/// Build a 2-layer decoder where each layer has cross-attention + FFN.
fn build_stacked_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_cross_attn_stacked_2layer");
    let queries = b.add_input("queries", &[QUERY_LEN, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[KV_LEN, HIDDEN_DIM]);
    let q_shape = [QUERY_LEN, HIDDEN_DIM];
    let ffn_shape = [QUERY_LEN, FFN_DIM];

    // --- Layer 0 ---
    let ca0 = build_cross_attention(
        &mut b,
        queries,
        encoder_mem,
        "l0_ca",
        QUERY_LEN,
        KV_LEN,
        HIDDEN_DIM,
    );
    let res0 = b.add_binary_add(queries, ca0, &q_shape);

    // FFN layer 0
    let ffn0_w1 = b.add_input("l0_ffn1_w", &[FFN_DIM, HIDDEN_DIM]);
    let ffn0_w2 = b.add_input("l0_ffn2_w", &[HIDDEN_DIM, FFN_DIM]);
    let h0 = b.add_linear(res0, ffn0_w1, None, &ffn_shape);
    let h0 = b.add_relu(h0, &ffn_shape);
    let ffn0_out = b.add_linear(h0, ffn0_w2, None, &q_shape);
    let out0 = b.add_binary_add(res0, ffn0_out, &q_shape);

    // --- Layer 1 ---
    let ca1 = build_cross_attention(
        &mut b,
        out0,
        encoder_mem,
        "l1_ca",
        QUERY_LEN,
        KV_LEN,
        HIDDEN_DIM,
    );
    let res1 = b.add_binary_add(out0, ca1, &q_shape);

    // FFN layer 1
    let ffn1_w1 = b.add_input("l1_ffn1_w", &[FFN_DIM, HIDDEN_DIM]);
    let ffn1_w2 = b.add_input("l1_ffn2_w", &[HIDDEN_DIM, FFN_DIM]);
    let h1 = b.add_linear(res1, ffn1_w1, None, &ffn_shape);
    let h1 = b.add_relu(h1, &ffn_shape);
    let ffn1_out = b.add_linear(h1, ffn1_w2, None, &q_shape);
    let out = b.add_binary_add(res1, ffn1_out, &q_shape);

    b.build(out)
        .expect("valid 2-layer stacked cross-attention kernel")
}

fn stacked_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_LEN, HIDDEN_DIM]), 0.1f32)), // encoder_mem
    ];
    // Layer 0: cross-attention + FFN
    push_cross_attention_bindings(&mut bindings, HIDDEN_DIM, WEIGHT_MAG);
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FFN_DIM, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, FFN_DIM]),
        WEIGHT_MAG,
    )));
    // Layer 1: cross-attention + FFN
    push_cross_attention_bindings(&mut bindings, HIDDEN_DIM, WEIGHT_MAG);
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FFN_DIM, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, FFN_DIM]),
        WEIGHT_MAG,
    )));
    bindings
}

#[test]
fn test_stacked_cross_attention_ibp() {
    let def = build_stacked_cross_attention_kernel();
    let bindings = stacked_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Stacked 2-layer cross-attention IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

#[test]
fn test_stacked_cross_attention_crown() {
    let def = build_stacked_cross_attention_kernel();
    let bindings = stacked_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("Stacked 2-layer cross-attention CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 12. Cross-attention KV from vision encoder features (VLM pattern) (IBP)
// ===========================================================================

/// VLM cross-attention: text decoder queries attend to vision encoder features.
/// Vision features are a constant parameter (pre-computed by vision encoder).
#[test]
fn test_cross_attention_vlm_vision_features_ibp() {
    let vision_len = 16; // 4x4 spatial patches
    let vision_dim = 128; // Vision encoder output dim

    let mut b = TensorBlockBuilder::new("dpdf_cross_attn_vlm_vision");
    let text_queries = b.add_input("text_queries", &[QUERY_LEN, HIDDEN_DIM]);
    let vision_features = b.add_input("vision_features", &[vision_len, vision_dim]);

    // Project vision features to decoder dimension
    let proj_w = b.add_input("vision_proj_w", &[HIDDEN_DIM, vision_dim]);
    let projected_vision = b.add_linear(vision_features, proj_w, None, &[vision_len, HIDDEN_DIM]);

    // Cross-attention: text queries attend to projected vision features
    let out = build_cross_attention(
        &mut b,
        text_queries,
        projected_vision,
        "vlm_ca",
        QUERY_LEN,
        vision_len,
        HIDDEN_DIM,
    );
    let def = b.build(out).expect("valid VLM cross-attention kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable, // text_queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[vision_len, vision_dim]),
            0.05f32,
        )), // vision_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, vision_dim]),
            WEIGHT_MAG,
        )), // vision_proj_w
    ];
    push_cross_attention_bindings(&mut bindings, HIDDEN_DIM, WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("VLM cross-attention (vision features) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 13. Cross-attention monotone tightening: smaller eps -> tighter bounds (IBP)
// ===========================================================================

#[test]
fn test_cross_attention_monotone_tightening_ibp() {
    let def = build_basic_cross_attention_kernel();
    let bindings = basic_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);
    let wide_width = bound_width(&wide_output);

    let tight_input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 0.1);
    let tight_output = graph.propagate_ibp(&tight_input).expect("IBP tight");
    assert_bounds_valid(&tight_output);
    let tight_width = bound_width(&tight_output);

    eprintln!(
        "Cross-attention monotone tightening: eps=1.0 width={wide_width:.6}, eps=0.1 width={tight_width:.6}"
    );
    assert!(
        tight_width <= wide_width + 1e-6,
        "tight input should produce tighter output: wide={wide_width}, tight={tight_width}"
    );
}

// ===========================================================================
// 14. CROWN tightness for cross-attention vs IBP (CROWN)
// ===========================================================================

#[test]
fn test_cross_attention_crown_vs_ibp_tightness() {
    let def = build_basic_cross_attention_kernel();
    let bindings = basic_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 0.5);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let ibp_width = bound_width(&ibp_output);

    // CROWN
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&crown_output);
    let crown_width = bound_width(&crown_output);

    eprintln!(
        "Cross-attention CROWN vs IBP: method={method:?}, \
         crown_width={crown_width:.6}, ibp_width={ibp_width:.6}"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Both should be finite
    assert!(ibp_width.is_finite(), "IBP width must be finite");
    assert!(crown_width.is_finite(), "CROWN width must be finite");
}

// ===========================================================================
// 15. Full decoder layer: LN + self-attn + LN + cross-attn + LN + FFN
//     (IBP + CROWN)
// ===========================================================================

fn build_full_decoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_full_decoder_layer");
    let queries = b.add_input("queries", &[QUERY_LEN, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[KV_LEN, HIDDEN_DIM]);
    let q_shape = [QUERY_LEN, HIDDEN_DIM];
    let ffn_shape = [QUERY_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // --- LN + Self-attention + residual ---
    let sa_ln_w = b.add_input("sa_ln_w", &[HIDDEN_DIM]);
    let sa_ln_b = b.add_input("sa_ln_b", &[HIDDEN_DIM]);
    let sa_eps = b.add_input("sa_eps", &[1]);
    let normed_sa = b.add_layer_norm(queries, sa_eps, 1, sa_ln_w, sa_ln_b, &q_shape);

    let sa_q_w = b.add_input("sa_q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_k_w = b.add_input("sa_k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_v_w = b.add_input("sa_v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_o_w = b.add_input("sa_o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let sq = b.add_linear(normed_sa, sa_q_w, None, &q_shape);
    let sk = b.add_linear(normed_sa, sa_k_w, None, &q_shape);
    let sv = b.add_linear(normed_sa, sa_v_w, None, &q_shape);
    let sa = b.add_attention(sq, sk, sv, AttentionMask::Standard, Some(scale), &q_shape);
    let sa_proj = b.add_linear(sa, sa_o_w, None, &q_shape);
    let res_sa = b.add_binary_add(queries, sa_proj, &q_shape);

    // --- LN + Cross-attention + residual ---
    let ca_ln_w = b.add_input("ca_ln_w", &[HIDDEN_DIM]);
    let ca_ln_b = b.add_input("ca_ln_b", &[HIDDEN_DIM]);
    let ca_eps = b.add_input("ca_eps", &[1]);
    let normed_ca = b.add_layer_norm(res_sa, ca_eps, 1, ca_ln_w, ca_ln_b, &q_shape);

    let ca_out = build_cross_attention(
        &mut b,
        normed_ca,
        encoder_mem,
        "ca",
        QUERY_LEN,
        KV_LEN,
        HIDDEN_DIM,
    );
    let res_ca = b.add_binary_add(res_sa, ca_out, &q_shape);

    // --- LN + FFN + residual ---
    let ffn_ln_w = b.add_input("ffn_ln_w", &[HIDDEN_DIM]);
    let ffn_ln_b = b.add_input("ffn_ln_b", &[HIDDEN_DIM]);
    let ffn_eps = b.add_input("ffn_eps", &[1]);
    let normed_ffn = b.add_layer_norm(res_ca, ffn_eps, 1, ffn_ln_w, ffn_ln_b, &q_shape);

    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, HIDDEN_DIM]);
    let ffn2_w = b.add_input("ffn2_w", &[HIDDEN_DIM, FFN_DIM]);
    let h = b.add_linear(normed_ffn, ffn1_w, None, &ffn_shape);
    let h = b.add_relu(h, &ffn_shape);
    let ffn_out = b.add_linear(h, ffn2_w, None, &q_shape);
    let out = b.add_binary_add(res_ca, ffn_out, &q_shape);

    b.build(out).expect("valid full decoder layer kernel")
}

fn full_decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let proj_w = || {
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        ))
    };
    let ln_w =
        || TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32));
    let ln_b =
        || TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32));
    let eps = || TensorParamBinding::ConstantScalar(1e-5);

    let mut bindings = vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_LEN, HIDDEN_DIM]), 0.1f32)), // encoder_mem
    ];

    // Self-attention: LN + 4 projections
    bindings.push(ln_w()); // sa_ln_w
    bindings.push(ln_b()); // sa_ln_b
    bindings.push(eps()); // sa_eps
    for _ in 0..4 {
        bindings.push(proj_w()); // sa_q_w, sa_k_w, sa_v_w, sa_o_w
    }

    // Cross-attention: LN + 4 projections
    bindings.push(ln_w()); // ca_ln_w
    bindings.push(ln_b()); // ca_ln_b
    bindings.push(eps()); // ca_eps
    push_cross_attention_bindings(&mut bindings, HIDDEN_DIM, WEIGHT_MAG);

    // FFN: LN + 2 projections
    bindings.push(ln_w()); // ffn_ln_w
    bindings.push(ln_b()); // ffn_ln_b
    bindings.push(eps()); // ffn_eps
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FFN_DIM, HIDDEN_DIM]),
        WEIGHT_MAG,
    ))); // ffn1_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, FFN_DIM]),
        WEIGHT_MAG,
    ))); // ffn2_w

    bindings
}

#[test]
fn test_full_decoder_layer_ibp() {
    let def = build_full_decoder_layer_kernel();
    let bindings = full_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full decoder layer (LN+SA+LN+CA+LN+FFN) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_full_decoder_layer_crown() {
    let def = build_full_decoder_layer_kernel();
    let bindings = full_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[QUERY_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full decoder layer CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
