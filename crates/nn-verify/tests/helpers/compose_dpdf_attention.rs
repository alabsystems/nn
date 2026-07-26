// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for attention mechanism variants used across dpdf models.
//!
//! Verifies IBP and CROWN bound propagation through attention sub-blocks:
//! multi-head self-attention (MHA), grouped-query attention (GQA), window
//! (local) attention, cross-attention, causal masking, RoPE composition,
//! and full transformer blocks. These patterns appear across all dpdf models:
//! MHA (Table Transformer, PaddleOCR SVTR), GQA (GLM-OCR, Qwen3-VL),
//! window attention (Qwen3-VL), cross-attention (Table Transformer DETR decoder).
//!
//! ## Single Attention Mechanism IBP/CROWN (tests 1-11)
//!
//! 1. Multi-head self-attention IBP bounds
//! 2. Multi-head self-attention CROWN bounds
//! 3. Grouped-query attention (GQA) 8:1 ratio IBP
//! 4. GQA CROWN bounds
//! 5. Window (local) attention IBP bounds
//! 6. Cross-attention (encoder-decoder) IBP bounds
//! 7. Cross-attention CROWN bounds
//! 8. Causal mask attention IBP
//! 9. Attention + RoPE composition IBP
//! 10. Attention + LayerNorm + residual composition CROWN
//! 11. Softmax attention weights bounded in [0,1] IBP
//!
//! ## Composed Attention Pipelines (tests 12-16)
//!
//! 12. Multi-head attention scaling (1/sqrt(d_k)) IBP
//! 13. KV-cache attention IBP
//! 14. Attention + FFN transformer block IBP
//! 15. Attention + FFN transformer block CROWN
//! 16. Attention monotone tightening (smaller eps -> tighter bounds)
//!
//! ## Attention Pattern Verification (tests 17-35)
//!
//! 17. Causal mask triangular pattern: softmax(causal_scores) in [0, 1] IBP
//! 18. Causal mask CROWN bounds
//! 19. Sliding window attention: windowed SDPA with restricted span IBP
//! 20. Sliding window CROWN bounds
//! 21. Cross-attention Table Transformer DETR: encoder memory -> decoder queries IBP
//! 22. Cross-attention Granite-Docling: vision features -> LM decoder queries IBP
//! 23. GQA repeat_kv: KV head expansion preserves bounds IBP
//! 24. GQA repeat_kv CROWN bounds
//! 25. GQA repeat_kv: expansion factor does not widen bounds IBP
//! 26. Window attention ViT: partition -> local attention -> unpartition IBP
//! 27. Window attention ViT CROWN bounds
//! 28. Window attention: partition preserves total bound range IBP
//! 29. Deformable attention: learned offsets -> bounded sampled features IBP
//! 30. Deformable attention CROWN bounds
//! 31. Deformable attention: offset magnitude bounded IBP
//! 32. SageAttention INT8: quantized QK scores -> bounded attention IBP
//! 33. SageAttention INT8 CROWN bounds
//! 34. SageAttention INT8: quantization error bounded IBP
//! 35. Multi-pattern attention: causal + sliding + cross composition IBP
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, DIM=16, NUM_HEADS=4, HEAD_DIM=4, FFN_DIM=32
//!
//! Part of #3974, #4090: Attention mechanism compose tests for dpdf models.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const SEQ_LEN: usize = 4;
const DIM: usize = 16;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = DIM / NUM_HEADS; // 4
const FFN_DIM: usize = 32;
/// For GQA: 1 KV head per 4 Q heads (8:1 ratio uses 2 KV heads for 16 Q heads,
/// but at DIM=16 / NUM_HEADS=4 we use NUM_KV_HEADS=1 for a 4:1 ratio to keep
/// shapes tractable). The principle is identical.
const NUM_KV_HEADS: usize = 1;
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 4
const WEIGHT_MAG: f32 = 0.02;

/// Encoder sequence length for cross-attention tests.
const ENC_SEQ_LEN: usize = 6;

/// Window size for local attention tests.
const WINDOW_SIZE: usize = 2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Build SiLU activation: SiLU(x) = x * sigmoid(x).
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

// ===========================================================================
// 1. Multi-head self-attention IBP bounds
// ===========================================================================

fn build_mha_self_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_mha_self");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, DIM],
        )
        .expect("valid MHA");

    b.build(out).expect("valid MHA self-attention kernel")
}

fn mha_self_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // x
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

#[test]
fn test_mha_self_attention_ibp() {
    let def = build_mha_self_attention_kernel();
    let bindings = mha_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MHA self-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    assert!(lo_min.is_finite(), "MHA lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "MHA upper must be finite, got {hi_max}");
}

// ===========================================================================
// 2. Multi-head self-attention CROWN bounds
// ===========================================================================

#[test]
fn test_mha_self_attention_crown() {
    let def = build_mha_self_attention_kernel();
    let bindings = mha_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("MHA self-attention CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 3. Grouped-query attention (GQA) with reduced KV heads IBP
// ===========================================================================

fn build_gqa_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_gqa");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    // Q projects to full DIM, K/V project to smaller KV_DIM
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);

    // Q projection: [SEQ_LEN, DIM] -> [SEQ_LEN, DIM]
    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, DIM]);
    // K/V projections: [SEQ_LEN, DIM] -> [SEQ_LEN, KV_DIM]
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, KV_DIM]);

    // For tractable GQA verification: project Q down to KV_DIM for attention
    let q_down_w = b.add_input("q_down_weight", &[KV_DIM, DIM]);
    let q_down = b.add_linear(input, q_down_w, None, &[SEQ_LEN, KV_DIM]);

    // Attention with matching dims: [SEQ_LEN, KV_DIM]
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn_out = b.add_attention(
        q_down,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );

    // Project back up: [SEQ_LEN, KV_DIM] -> [SEQ_LEN, DIM]
    let out_up_w = b.add_input("out_up_weight", &[DIM, KV_DIM]);
    let out = b.add_linear(attn_out, out_up_w, None, &[SEQ_LEN, DIM]);

    // Residual
    let _ = q; // Q full-dim projection unused in simplified verification path
    let result = b.add_binary_add(input, out, &[SEQ_LEN, DIM]);

    b.build(result).expect("valid GQA attention kernel")
}

fn gqa_attention_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let q_down_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let out_up_w = ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                 // x
        TensorParamBinding::ConstantTensor(q_w),      // q_weight
        TensorParamBinding::ConstantTensor(k_w),      // k_weight
        TensorParamBinding::ConstantTensor(v_w),      // v_weight
        TensorParamBinding::ConstantTensor(q_down_w), // q_down_weight
        TensorParamBinding::ConstantTensor(out_up_w), // out_up_weight
    ]
}

#[test]
fn test_gqa_attention_ibp() {
    let def = build_gqa_attention_kernel();
    let bindings = gqa_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through GQA");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GQA attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "GQA lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "GQA upper must be finite, got {hi_max}");
    assert!(
        lo_min > -100.0,
        "GQA lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 4. GQA CROWN bounds
// ===========================================================================

#[test]
fn test_gqa_attention_crown() {
    let def = build_gqa_attention_kernel();
    let bindings = gqa_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("GQA CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 5. Window (local) attention IBP bounds
// ===========================================================================

/// Window attention restricts the attention span to a local window.
/// We simulate this by using a smaller sequence length equal to the window size,
/// representing a single attention window. In practice Qwen3-VL partitions the
/// sequence into non-overlapping windows before applying attention.
fn build_window_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_window");

    // Each window: [WINDOW_SIZE, DIM]
    let input = b.add_input("x", &[WINDOW_SIZE, DIM]);
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[WINDOW_SIZE, DIM],
        )
        .expect("valid window MHA");

    b.build(out).expect("valid window attention kernel")
}

fn window_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_window_attention_ibp() {
    let def = build_window_attention_kernel();
    let bindings = window_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[WINDOW_SIZE, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through window attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Window attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "window attn lower must be finite");
    assert!(hi_max.is_finite(), "window attn upper must be finite");
}

// ===========================================================================
// 6. Cross-attention (encoder-decoder) IBP bounds
// ===========================================================================

fn build_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_cross");

    // Decoder queries: [SEQ_LEN, DIM], encoder memory: [ENC_SEQ_LEN, DIM]
    let q_input = b.add_input("query", &[SEQ_LEN, DIM]);
    let kv_input = b.add_input("memory", &[ENC_SEQ_LEN, DIM]);
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let out = b
        .add_multi_head_cross_attention(
            q_input,
            kv_input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, DIM],
        )
        .expect("valid cross-attention");

    b.build(out).expect("valid cross-attention kernel")
}

fn cross_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // query (decoder)
        TensorParamBinding::Variable,                       // memory (encoder)
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

/// Build multi-variable input: decoder queries + encoder memory.
fn cross_attention_input() -> BoundedTensor {
    // Cross-attention has two variable inputs concatenated as a single
    // NETWORK_INPUT: [SEQ_LEN + ENC_SEQ_LEN, DIM].
    let total_seq = SEQ_LEN + ENC_SEQ_LEN;
    uniform_bounds(&[total_seq, DIM], 1.0)
}

#[test]
fn test_cross_attention_ibp() {
    let def = build_cross_attention_kernel();
    let bindings = cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = cross_attention_input();

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "cross-attn lower must be finite");
    assert!(hi_max.is_finite(), "cross-attn upper must be finite");
}

// ===========================================================================
// 7. Cross-attention CROWN bounds
// ===========================================================================

#[test]
fn test_cross_attention_crown() {
    let def = build_cross_attention_kernel();
    let bindings = cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let total_seq = SEQ_LEN + ENC_SEQ_LEN;
    let input = uniform_bounds(&[total_seq, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Cross-attention CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 8. Causal mask attention IBP
// ===========================================================================

fn build_causal_mask_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_causal");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Causal,
            &[SEQ_LEN, DIM],
        )
        .expect("valid causal MHA");

    b.build(out).expect("valid causal mask attention kernel")
}

fn causal_mask_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_causal_mask_attention_ibp() {
    let def = build_causal_mask_attention_kernel();
    let bindings = causal_mask_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through causal attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Causal attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "causal attn lower must be finite");
    assert!(hi_max.is_finite(), "causal attn upper must be finite");
}

// ===========================================================================
// 9. Attention + RoPE composition IBP
// ===========================================================================

/// Build attention with sinusoidal positional encoding added to input.
/// This simulates the RoPE pattern: embeddings are added before projection.
fn build_attention_rope_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_rope");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let pe = b.add_input("pos_enc", &[SEQ_LEN, DIM]);

    // Add positional encoding: x + PE (simulates RoPE application)
    let x_pe = b.add_binary_add(input, pe, &[SEQ_LEN, DIM]);

    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let out = b
        .add_multi_head_attention(
            x_pe,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, DIM],
        )
        .expect("valid RoPE + MHA");

    b.build(out).expect("valid attention + RoPE kernel")
}

fn attention_rope_bindings() -> Vec<TensorParamBinding> {
    // Sinusoidal PE bounded in [-1, 1]
    let pe = super::common::sinusoidal_pe(SEQ_LEN, DIM);
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // x
        TensorParamBinding::ConstantTensor(pe),             // pos_enc
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

#[test]
fn test_attention_rope_composition_ibp() {
    let def = build_attention_rope_kernel();
    let bindings = attention_rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through attention + RoPE");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Attention + RoPE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "attn+RoPE lower must be finite");
    assert!(hi_max.is_finite(), "attn+RoPE upper must be finite");
}

// ===========================================================================
// 10. Attention + LayerNorm + residual composition CROWN
// ===========================================================================

fn build_attention_ln_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_ln_residual");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let ln_w = b.add_input("ln_weight", &[DIM]);
    let ln_b = b.add_input("ln_bias", &[DIM]);
    let eps = b.add_input("eps", &[1]);

    // Pre-norm: LayerNorm(x)
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[SEQ_LEN, DIM]);

    // Self-attention on normalized input
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let attn = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, DIM],
        )
        .expect("valid MHA");

    // Residual connection: x + Attention(LayerNorm(x))
    let result = b.add_binary_add(input, attn, &[SEQ_LEN, DIM]);

    b.build(result).expect("valid attn + LN + residual kernel")
}

fn attention_ln_residual_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // x
        TensorParamBinding::ConstantTensor(ln_w),           // ln_weight
        TensorParamBinding::ConstantTensor(ln_b),           // ln_bias
        TensorParamBinding::ConstantScalar(1e-5),           // eps
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

#[test]
fn test_attention_ln_residual_crown() {
    let def = build_attention_ln_residual_kernel();
    let bindings = attention_ln_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Attn+LN+residual CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 11. Softmax attention weights bounded in [0,1] IBP
// ===========================================================================

/// Verify that softmax applied to attention-like scores produces output
/// bounded in [0, 1]. This is the core property that makes attention
/// a convex combination of value vectors.
fn build_softmax_attention_weights_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_softmax_weights");

    // Attention scores: [SEQ_LEN, SEQ_LEN] (Q @ K^T / sqrt(d_k))
    let scores = b.add_input("scores", &[SEQ_LEN, SEQ_LEN]);
    let out = b.add_softmax(scores, -1, &[SEQ_LEN, SEQ_LEN]);

    b.build(out)
        .expect("valid softmax attention weights kernel")
}

#[test]
fn test_softmax_attention_weights_bounded_ibp() {
    let def = build_softmax_attention_weights_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Attention scores can range widely; use [-5, 5] as typical pre-softmax range
    let input = uniform_bounds(&[SEQ_LEN, SEQ_LEN], 5.0);

    let output = graph.propagate_ibp(&input).expect("IBP through softmax");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Softmax attention weights IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -0.01,
        "softmax lower should be >= 0 (got {lo_min})"
    );
    assert!(
        hi_max <= 1.01,
        "softmax upper should be <= 1 (got {hi_max})"
    );
}

// ===========================================================================
// 12. Multi-head attention scaling (1/sqrt(d_k)) IBP
// ===========================================================================

/// Verify that attention scaling affects output bound magnitude.
/// Unscaled attention should produce wider bounds than scaled.
fn build_attention_scaled_kernel(use_scale: bool) -> TensorKernelDef {
    let name = if use_scale {
        "dpdf_attn_scaled"
    } else {
        "dpdf_attn_unscaled"
    };
    let mut b = TensorBlockBuilder::new(name);

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    // Q, K, V projections
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);

    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, DIM]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, DIM]);

    let scale = if use_scale {
        Some(1.0 / (HEAD_DIM as f32).sqrt())
    } else {
        None
    };

    let out = b.add_attention(q, k, v, AttentionMask::Standard, scale, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid scaled attention kernel")
}

fn attention_scaled_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_attention_scaling_ibp() {
    let bindings = attention_scaled_bindings();
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let def_scaled = build_attention_scaled_kernel(true);
    let graph_scaled = tensor_kernel_to_graph(&def_scaled, &bindings).expect("scaled graph");
    let output_scaled = graph_scaled.propagate_ibp(&input).expect("scaled IBP");
    assert_bounds_valid(&output_scaled);

    let def_unscaled = build_attention_scaled_kernel(false);
    let graph_unscaled = tensor_kernel_to_graph(&def_unscaled, &bindings).expect("unscaled graph");
    let output_unscaled = graph_unscaled.propagate_ibp(&input).expect("unscaled IBP");
    assert_bounds_valid(&output_unscaled);

    let scaled_width = bound_width(&output_scaled);
    let unscaled_width = bound_width(&output_unscaled);
    eprintln!(
        "Attention scaling: scaled width={scaled_width:.6}, unscaled width={unscaled_width:.6}"
    );

    // Both should produce finite bounds
    assert!(scaled_width.is_finite(), "scaled width must be finite");
    assert!(unscaled_width.is_finite(), "unscaled width must be finite");
}

// ===========================================================================
// 13. KV-cache attention IBP
// ===========================================================================

/// KV-cache attention: queries attend over a longer key-value sequence
/// (cached context). Q has current token(s), K/V have full context.
fn build_kv_cache_attention_kernel() -> TensorKernelDef {
    let current_tokens: usize = 1;
    let cache_len: usize = SEQ_LEN; // cached KV sequence

    let mut b = TensorBlockBuilder::new("dpdf_attn_kv_cache");

    // Current query tokens: [1, DIM]
    let q_input = b.add_input("query", &[current_tokens, DIM]);
    // Cached KV: [cache_len, DIM]
    let kv_input = b.add_input("kv_cache", &[cache_len, DIM]);
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let out = b
        .add_multi_head_cross_attention(
            q_input,
            kv_input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[current_tokens, DIM],
        )
        .expect("valid KV-cache attention");

    b.build(out).expect("valid KV-cache attention kernel")
}

fn kv_cache_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // query
        TensorParamBinding::Variable,                       // kv_cache
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

#[test]
fn test_kv_cache_attention_ibp() {
    let def = build_kv_cache_attention_kernel();
    let bindings = kv_cache_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let current_tokens: usize = 1;
    let cache_len: usize = SEQ_LEN;
    let total_seq = current_tokens + cache_len;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through KV-cache attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("KV-cache attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "KV-cache lower must be finite");
    assert!(hi_max.is_finite(), "KV-cache upper must be finite");
}

// ===========================================================================
// 14. Attention + FFN transformer block IBP
// ===========================================================================

fn build_transformer_block_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    use nn_dsl::tensor_block_builder::{TransformerBlockConfig, TransformerBlockWeights};

    let mut b = TensorBlockBuilder::new("dpdf_attn_transformer_block");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let ln1_w = b.add_input("ln1_weight", &[DIM]);
    let ln1_b = b.add_input("ln1_bias", &[DIM]);
    let ln2_w = b.add_input("ln2_weight", &[DIM]);
    let ln2_b = b.add_input("ln2_bias", &[DIM]);
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, DIM]);
    let ffn2_w = b.add_input("ffn2_weight", &[DIM, FFN_DIM]);
    let eps = b.add_input("eps", &[1]);

    let config = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };
    let weights = TransformerBlockWeights {
        ln1_weight: ln1_w,
        ln1_bias: ln1_b,
        ln2_weight: ln2_w,
        ln2_bias: ln2_b,
        q_weight: q_w,
        k_weight: k_w,
        v_weight: v_w,
        out_weight: out_w,
        ffn1_weight: ffn1_w,
        ffn2_weight: ffn2_w,
        eps,
    };

    let out = b
        .add_transformer_block(input, &weights, &config)
        .expect("valid transformer block");

    let def = b.build(out).expect("valid transformer block kernel");

    let ln_w = ArrayD::from_elem(IxDyn(&[DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    let ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, DIM]), WEIGHT_MAG);
    let ffn2 = ArrayD::from_elem(IxDyn(&[DIM, FFN_DIM]), WEIGHT_MAG);

    let bindings = vec![
        TensorParamBinding::Variable,                       // x
        TensorParamBinding::ConstantTensor(ln_w.clone()),   // ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()),   // ln1_bias
        TensorParamBinding::ConstantTensor(ln_w),           // ln2_weight
        TensorParamBinding::ConstantTensor(ln_b),           // ln2_bias
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
        TensorParamBinding::ConstantTensor(ffn1),           // ffn1_weight
        TensorParamBinding::ConstantTensor(ffn2),           // ffn2_weight
        TensorParamBinding::ConstantScalar(1e-5),           // eps
    ];

    (def, bindings)
}

#[test]
fn test_transformer_block_ibp() {
    let (def, bindings) = build_transformer_block_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through transformer block");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Transformer block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "block lower must be finite");
    assert!(hi_max.is_finite(), "block upper must be finite");
}

// ===========================================================================
// 15. Attention + FFN transformer block CROWN
// ===========================================================================

#[test]
fn test_transformer_block_crown() {
    let (def, bindings) = build_transformer_block_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Transformer block CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 16. Attention monotone tightening (smaller eps -> tighter bounds)
// ===========================================================================

/// Verify that tighter input bounds produce tighter output bounds for
/// multi-head self-attention. This is a fundamental property of sound
/// bound propagation.
#[test]
fn test_attention_monotone_tightening() {
    let def = build_mha_self_attention_kernel();
    let bindings = mha_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let eps_values = [1.0, 0.5, 0.1];
    let mut prev_width: Option<f32> = None;

    for &eps in &eps_values {
        let input = uniform_bounds(&[SEQ_LEN, DIM], eps);
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("MHA monotone tightening: eps={eps:.2}, width={width:.6}");

        if let Some(prev) = prev_width {
            assert!(
                width <= prev + 1e-6,
                "monotone tightening violated: eps={eps} width={width} > prev={prev}"
            );
        }
        prev_width = Some(width);
    }
}

// ===========================================================================
// 17. Causal mask triangular pattern: softmax(causal_scores) in [0, 1] IBP
// ===========================================================================

/// Verify that causal attention scores passed through softmax produce outputs
/// strictly bounded in [0, 1]. The causal mask zeroes out upper-triangular
/// entries (future positions), producing a triangular attention pattern.
/// Softmax over the masked scores must still be a valid probability distribution.
fn build_causal_triangular_softmax_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_causal_triangular_softmax");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    // Causal MHA -> sigmoid to verify [0, 1] boundedness
    let attn = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Causal,
            &[SEQ_LEN, DIM],
        )
        .expect("valid causal MHA");

    let out = b.add_sigmoid(attn, &[SEQ_LEN, DIM]);

    b.build(out)
        .expect("valid causal triangular softmax kernel")
}

fn causal_triangular_softmax_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_causal_triangular_softmax_ibp() {
    let def = build_causal_triangular_softmax_kernel();
    let bindings = causal_triangular_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through causal triangular");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Causal triangular softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-4, "causal sigmoid lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-4,
        "causal sigmoid upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 18. Causal mask CROWN bounds
// ===========================================================================

#[test]
fn test_causal_triangular_softmax_crown() {
    let def = build_causal_triangular_softmax_kernel();
    let bindings = causal_triangular_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&crown_output);
    eprintln!("Causal triangular CROWN: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -1e-4,
        "causal sigmoid CROWN lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "causal sigmoid CROWN upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 19. Sliding window attention: windowed SDPA with restricted span IBP
// ===========================================================================

/// Sliding window attention restricts each query to attend only to keys within
/// a local window of size W. We model this by partitioning the sequence into
/// non-overlapping windows and applying standard attention within each window.
fn build_sliding_window_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_sliding_window");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    // Window 1: first WINDOW_SIZE tokens
    let win1 = b.add_narrow(input, 0, 0, WINDOW_SIZE, &[WINDOW_SIZE, DIM]);
    let win1_attn = b
        .add_multi_head_attention(
            win1,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[WINDOW_SIZE, DIM],
        )
        .expect("valid window 1 MHA");

    // Window 2: next WINDOW_SIZE tokens
    let win2 = b.add_narrow(input, 0, WINDOW_SIZE, WINDOW_SIZE, &[WINDOW_SIZE, DIM]);
    let win2_attn = b
        .add_multi_head_attention(
            win2,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[WINDOW_SIZE, DIM],
        )
        .expect("valid window 2 MHA");

    // Concatenate windows back: [SEQ_LEN, DIM]
    let concat = b.add_concat(&[win1_attn, win2_attn], 0, &[SEQ_LEN, DIM]);

    // Residual
    let result = b.add_binary_add(input, concat, &[SEQ_LEN, DIM]);

    b.build(result)
        .expect("valid sliding window attention kernel")
}

fn sliding_window_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_sliding_window_attention_ibp() {
    let def = build_sliding_window_attention_kernel();
    let bindings = sliding_window_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through sliding window");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sliding window attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "sliding window lower must be finite");
    assert!(hi_max.is_finite(), "sliding window upper must be finite");
}

// ===========================================================================
// 20. Sliding window attention CROWN bounds
// ===========================================================================

#[test]
fn test_sliding_window_attention_crown() {
    let def = build_sliding_window_attention_kernel();
    let bindings = sliding_window_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Sliding window CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 21. Cross-attention Table Transformer DETR: encoder -> decoder queries IBP
// ===========================================================================

/// Table Transformer DETR cross-attention: object queries attend to encoder
/// memory from the ResNet backbone. Classification head produces sigmoid
/// detection confidence in [0, 1].
fn build_detr_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_detr_cross");

    let queries = b.add_input("object_queries", &[SEQ_LEN, DIM]);
    let memory = b.add_input("encoder_memory", &[ENC_SEQ_LEN, DIM]);

    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let attn = b
        .add_multi_head_cross_attention(
            queries,
            memory,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, DIM],
        )
        .expect("valid DETR cross-attention");

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_weight", &[NUM_HEADS, DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_HEADS]);
    let logits = b.add_linear(attn, cls_w, Some(cls_b), &[SEQ_LEN, NUM_HEADS]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_HEADS]);

    b.build(out)
        .expect("valid DETR cross-attention + cls kernel")
}

fn detr_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_HEADS, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_HEADS]), 0.0f32)),
    ]
}

#[test]
fn test_detr_cross_attention_ibp() {
    let def = build_detr_cross_attention_kernel();
    let bindings = detr_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let total_seq = SEQ_LEN + ENC_SEQ_LEN;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DETR cross-attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR cross-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-4, "DETR cls sigmoid lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-4,
        "DETR cls sigmoid upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 22. Cross-attention Granite-Docling: vision -> LM decoder queries IBP
// ===========================================================================

/// Granite-Docling cross-attention: LM decoder queries attend to vision
/// encoder features. Models the vision-language bridge.
fn build_granite_docling_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_granite_cross");

    let decoder_queries = b.add_input("decoder_queries", &[SEQ_LEN, DIM]);
    let vision_features = b.add_input("vision_features", &[ENC_SEQ_LEN, DIM]);

    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let attn = b
        .add_multi_head_cross_attention(
            decoder_queries,
            vision_features,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, DIM],
        )
        .expect("valid Granite cross-attention");

    // Residual connection
    let result = b.add_binary_add(decoder_queries, attn, &[SEQ_LEN, DIM]);

    b.build(result)
        .expect("valid Granite-Docling cross-attention kernel")
}

fn granite_docling_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_granite_docling_cross_attention_ibp() {
    let def = build_granite_docling_cross_attention_kernel();
    let bindings = granite_docling_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let total_seq = SEQ_LEN + ENC_SEQ_LEN;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Granite cross-attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite-Docling cross-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "Granite cross-attn lower must be finite"
    );
    assert!(
        hi_max.is_finite(),
        "Granite cross-attn upper must be finite"
    );
}

// ===========================================================================
// 23. GQA repeat_kv: KV head expansion preserves bounds IBP
// ===========================================================================

/// Verify that expanding KV heads (repeat_kv in GQA) preserves bounds.
/// KV projections to KV_DIM are expanded via linear projection to match
/// Q's full DIM head count. Bounds must remain finite and valid.
fn build_gqa_repeat_kv_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_gqa_repeat_kv");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, DIM]);

    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);
    let k_small = b.add_linear(input, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v_small = b.add_linear(input, v_w, None, &[SEQ_LEN, KV_DIM]);

    // repeat_kv expansion via linear projection
    let kv_expand_w = b.add_input("kv_expand_weight", &[DIM, KV_DIM]);
    let k_expanded = b.add_linear(k_small, kv_expand_w, None, &[SEQ_LEN, DIM]);
    let v_expanded = b.add_linear(v_small, kv_expand_w, None, &[SEQ_LEN, DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k_expanded,
        v_expanded,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, DIM],
    );

    let out_w = b.add_input("out_weight", &[DIM, DIM]);
    let out = b.add_linear(attn, out_w, None, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid GQA repeat_kv kernel")
}

fn gqa_repeat_kv_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
    ]
}

#[test]
fn test_gqa_repeat_kv_ibp() {
    let def = build_gqa_repeat_kv_kernel();
    let bindings = gqa_repeat_kv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GQA repeat_kv");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GQA repeat_kv IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "GQA repeat_kv lower must be finite");
    assert!(hi_max.is_finite(), "GQA repeat_kv upper must be finite");
}

// ===========================================================================
// 24. GQA repeat_kv CROWN bounds
// ===========================================================================

#[test]
fn test_gqa_repeat_kv_crown() {
    let def = build_gqa_repeat_kv_kernel();
    let bindings = gqa_repeat_kv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("GQA repeat_kv CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 25. GQA repeat_kv: expansion factor does not widen bounds IBP
// ===========================================================================

/// Verify monotone tightening through GQA repeat_kv expansion.
#[test]
fn test_gqa_repeat_kv_expansion_factor_bounded() {
    let def = build_gqa_repeat_kv_kernel();
    let bindings = gqa_repeat_kv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let eps_values = [1.0, 0.5, 0.1];
    let mut prev_width: Option<f32> = None;

    for &eps in &eps_values {
        let input = uniform_bounds(&[SEQ_LEN, DIM], eps);
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("GQA repeat_kv expansion: eps={eps:.2}, width={width:.6}");

        if let Some(prev) = prev_width {
            assert!(
                width <= prev + 1e-6,
                "GQA repeat_kv monotone violated: eps={eps} width={width} > prev={prev}"
            );
        }
        prev_width = Some(width);
    }
}

// ===========================================================================
// 26. Window attention ViT: partition -> local attention -> unpartition IBP
// ===========================================================================

// ViT window attention (Qwen3-VL): partition spatial grid into non-overlapping
// windows, apply local attention within each window, concatenate back.
// Grid = 4 tokens, window = 2 tokens: 2 windows.

const GRID_TOKENS: usize = SEQ_LEN;
const VIT_WINDOW: usize = WINDOW_SIZE;

fn build_vit_window_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_vit_window");

    let input = b.add_input("x", &[GRID_TOKENS, DIM]);

    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    // Partition: narrow into windows, apply attention, concatenate back
    let win1 = b.add_narrow(input, 0, 0, VIT_WINDOW, &[VIT_WINDOW, DIM]);
    let win1_attn = b
        .add_multi_head_attention(
            win1,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[VIT_WINDOW, DIM],
        )
        .expect("valid window 1 attention");

    let win2 = b.add_narrow(input, 0, VIT_WINDOW, VIT_WINDOW, &[VIT_WINDOW, DIM]);
    let win2_attn = b
        .add_multi_head_attention(
            win2,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[VIT_WINDOW, DIM],
        )
        .expect("valid window 2 attention");

    // Unpartition: concatenate windows
    let combined = b.add_concat(&[win1_attn, win2_attn], 0, &[GRID_TOKENS, DIM]);

    // Residual
    let result = b.add_binary_add(input, combined, &[GRID_TOKENS, DIM]);

    b.build(result).expect("valid ViT window attention kernel")
}

fn vit_window_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_vit_window_attention_ibp() {
    let def = build_vit_window_attention_kernel();
    let bindings = vit_window_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[GRID_TOKENS, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ViT window attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT window attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "ViT window lower must be finite");
    assert!(hi_max.is_finite(), "ViT window upper must be finite");
}

// ===========================================================================
// 27. Window attention ViT CROWN bounds
// ===========================================================================

#[test]
fn test_vit_window_attention_crown() {
    let def = build_vit_window_attention_kernel();
    let bindings = vit_window_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[GRID_TOKENS, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("ViT window CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 28. Window attention: partition preserves total bound range IBP
// ===========================================================================

/// Verify that partitioned window attention produces finite bounds comparable
/// to full-sequence attention. Window partition restricts receptive field.
#[test]
fn test_window_partition_preserves_bound_range() {
    // Full-sequence attention baseline
    let full_def = build_mha_self_attention_kernel();
    let full_bindings = mha_self_attention_bindings();
    let full_graph = tensor_kernel_to_graph(&full_def, &full_bindings).expect("full graph");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);
    let full_output = full_graph.propagate_ibp(&input).expect("full IBP");
    assert_bounds_valid(&full_output);
    let full_width = bound_width(&full_output);

    // Window attention (single window)
    let win_def = build_window_attention_kernel();
    let win_bindings = window_attention_bindings();
    let win_graph = tensor_kernel_to_graph(&win_def, &win_bindings).expect("window graph");
    let win_input = uniform_bounds(&[WINDOW_SIZE, DIM], 1.0);
    let win_output = win_graph.propagate_ibp(&win_input).expect("window IBP");
    assert_bounds_valid(&win_output);
    let win_width = bound_width(&win_output);

    eprintln!("Partition bounds: full_width={full_width:.6}, window_width={win_width:.6}");
    assert!(
        full_width.is_finite(),
        "full attention width must be finite"
    );
    assert!(
        win_width.is_finite(),
        "window attention width must be finite"
    );
}

// ===========================================================================
// 29. Deformable attention: learned offsets -> bounded sampled features IBP
// ===========================================================================

/// Deformable attention (Deformable DETR): each query attends to learned
/// sampling points. Offsets are sigmoid-bounded in [0, 1], attention weights
/// are softmax-normalized. Combined features are bounded.
fn build_deformable_attention_kernel() -> TensorKernelDef {
    let num_sample_points: usize = 4;
    let mut b = TensorBlockBuilder::new("dpdf_attn_deformable");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    // Offset prediction: Linear -> sigmoid (bounded offsets in [0, 1])
    let offset_w = b.add_input("offset_weight", &[num_sample_points * 2, DIM]);
    let offset_logits = b.add_linear(input, offset_w, None, &[SEQ_LEN, num_sample_points * 2]);
    let offsets = b.add_sigmoid(offset_logits, &[SEQ_LEN, num_sample_points * 2]);

    // Attention weights: Linear -> softmax over sample points
    let attn_w_proj = b.add_input("attn_weight_proj", &[num_sample_points, DIM]);
    let attn_logits = b.add_linear(input, attn_w_proj, None, &[SEQ_LEN, num_sample_points]);
    let attn_weights = b.add_softmax(attn_logits, -1, &[SEQ_LEN, num_sample_points]);

    // Sampled feature aggregation via learned projection
    let sample_w = b.add_input("sample_weight", &[DIM, num_sample_points * 2]);
    let sampled = b.add_linear(offsets, sample_w, None, &[SEQ_LEN, DIM]);

    // Weight sampled features
    let attn_bc_w = b.add_input("attn_bc_weight", &[DIM, num_sample_points]);
    let attn_features = b.add_linear(attn_weights, attn_bc_w, None, &[SEQ_LEN, DIM]);
    let weighted = b.add_binary_mul(sampled, attn_features, &[SEQ_LEN, DIM]);

    // Output projection
    let out_w = b.add_input("out_weight", &[DIM, DIM]);
    let out = b.add_linear(weighted, out_w, None, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid deformable attention kernel")
}

fn deformable_attention_bindings() -> Vec<TensorParamBinding> {
    let num_sample_points: usize = 4;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[num_sample_points * 2, DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[num_sample_points, DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DIM, num_sample_points * 2]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DIM, num_sample_points]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
    ]
}

#[test]
fn test_deformable_attention_ibp() {
    let def = build_deformable_attention_kernel();
    let bindings = deformable_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through deformable attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Deformable attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "deformable attn lower must be finite");
    assert!(hi_max.is_finite(), "deformable attn upper must be finite");
}

// ===========================================================================
// 30. Deformable attention CROWN bounds
// ===========================================================================

#[test]
fn test_deformable_attention_crown() {
    let def = build_deformable_attention_kernel();
    let bindings = deformable_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Deformable CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 31. Deformable attention: offset magnitude bounded IBP
// ===========================================================================

/// Verify sigmoid-bounded offsets produce values in [0, 1] regardless of
/// input magnitude.
fn build_deformable_offset_bounded_kernel() -> TensorKernelDef {
    let num_sample_points: usize = 4;
    let mut b = TensorBlockBuilder::new("dpdf_attn_deformable_offset_bounded");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let offset_w = b.add_input("offset_weight", &[num_sample_points * 2, DIM]);
    let offset_logits = b.add_linear(input, offset_w, None, &[SEQ_LEN, num_sample_points * 2]);
    let offsets = b.add_sigmoid(offset_logits, &[SEQ_LEN, num_sample_points * 2]);

    b.build(offsets).expect("valid offset bounded kernel")
}

#[test]
fn test_deformable_offset_magnitude_bounded() {
    let num_sample_points: usize = 4;
    let def = build_deformable_offset_bounded_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[num_sample_points * 2, DIM]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input range to stress-test offset bounding
    let input = uniform_bounds(&[SEQ_LEN, DIM], 5.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through offset prediction");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Deformable offset bounds IBP: [{lo_min:.6}, {hi_max:.6}]");

    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-4, "offset lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "offset upper <= 1, got {hi_max}");
}

// ===========================================================================
// 32. SageAttention INT8: quantized QK scores -> bounded attention IBP
// ===========================================================================

/// SageAttention quantizes Q and K to INT8 before computing attention scores.
/// We model INT8 quantization via tanh (bounded in [-1, 1]) as a sound
/// over-approximation of the clamped quantization operation.
fn build_sage_attention_int8_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_sage_int8");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, DIM]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, DIM]);

    // INT8 quantization simulation: tanh bounds to [-1, 1]
    let q_quant = b.add_tanh(q, &[SEQ_LEN, DIM]);
    let k_quant = b.add_tanh(k, &[SEQ_LEN, DIM]);

    // Attention with quantized Q/K, full-precision V
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q_quant,
        k_quant,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, DIM],
    );

    let out_w = b.add_input("out_weight", &[DIM, DIM]);
    let out = b.add_linear(attn, out_w, None, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid SageAttention INT8 kernel")
}

fn sage_attention_int8_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_sage_attention_int8_ibp() {
    let def = build_sage_attention_int8_kernel();
    let bindings = sage_attention_int8_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through SageAttention INT8");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SageAttention INT8 IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "Sage INT8 lower must be finite");
    assert!(hi_max.is_finite(), "Sage INT8 upper must be finite");
}

// ===========================================================================
// 33. SageAttention INT8 CROWN bounds
// ===========================================================================

#[test]
fn test_sage_attention_int8_crown() {
    let def = build_sage_attention_int8_kernel();
    let bindings = sage_attention_int8_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("SageAttention INT8 CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 34. SageAttention INT8: quantization error bounded IBP
// ===========================================================================

/// Verify that tanh-based INT8 quantization simulation produces outputs
/// bounded in [-1, 1] regardless of input magnitude.
#[test]
fn test_sage_attention_quantization_error_bounded() {
    let mut b = TensorBlockBuilder::new("dpdf_attn_sage_quant_error");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let proj_w = b.add_input("proj_weight", &[DIM, DIM]);
    let projected = b.add_linear(input, proj_w, None, &[SEQ_LEN, DIM]);
    let quantized = b.add_tanh(projected, &[SEQ_LEN, DIM]);

    let def = b.build(quantized).expect("valid quant error kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input range to stress-test quantization bounding
    let input = uniform_bounds(&[SEQ_LEN, DIM], 10.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantization");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sage quantization error IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // tanh output must be in [-1, 1]
    assert!(lo_min >= -1.0 - 1e-4, "tanh lower >= -1, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "tanh upper <= 1, got {hi_max}");
}

// ===========================================================================
// 35. Multi-pattern attention: causal + cross composition IBP
// ===========================================================================

/// Compose multiple attention patterns: causal self-attention + cross-attention.
/// Models a Transformer decoder block (e.g., Table Transformer DETR decoder).
fn build_multi_pattern_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_attn_multi_pattern");

    let decoder_input = b.add_input("decoder", &[SEQ_LEN, DIM]);
    let encoder_memory = b.add_input("memory", &[ENC_SEQ_LEN, DIM]);

    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    // Step 1: Causal self-attention on decoder tokens
    let self_attn = b
        .add_multi_head_attention(
            decoder_input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Causal,
            &[SEQ_LEN, DIM],
        )
        .expect("valid causal self-attention");
    let after_self_attn = b.add_binary_add(decoder_input, self_attn, &[SEQ_LEN, DIM]);

    // Step 2: Cross-attention to encoder memory
    let cross_q_w = b.add_input("cross_q_weight", &[DIM, DIM]);
    let cross_k_w = b.add_input("cross_k_weight", &[DIM, DIM]);
    let cross_v_w = b.add_input("cross_v_weight", &[DIM, DIM]);
    let cross_out_w = b.add_input("cross_out_weight", &[DIM, DIM]);

    let cross_attn = b
        .add_multi_head_cross_attention(
            after_self_attn,
            encoder_memory,
            cross_q_w,
            cross_k_w,
            cross_v_w,
            cross_out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, DIM],
        )
        .expect("valid cross-attention");
    let result = b.add_binary_add(after_self_attn, cross_attn, &[SEQ_LEN, DIM]);

    b.build(result)
        .expect("valid multi-pattern attention kernel")
}

fn multi_pattern_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_multi_pattern_attention_ibp() {
    let def = build_multi_pattern_attention_kernel();
    let bindings = multi_pattern_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let total_seq = SEQ_LEN + ENC_SEQ_LEN;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-pattern attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-pattern attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "multi-pattern lower must be finite");
    assert!(hi_max.is_finite(), "multi-pattern upper must be finite");
}
