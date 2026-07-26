// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for GQA (Grouped-Query Attention) with KV-cache patterns
//! used in document VLMs (Qwen3-VL, FireRed-OCR, GLM-OCR).
//!
//! Verifies IBP and CROWN bound propagation through GQA attention sub-blocks
//! with KV-cache patterns: head grouping, cache append, sliding window,
//! output projection, cross-attention, prefill vs decode, RoPE integration,
//! cache memory layout, causal mask with offset, group ratio comparison,
//! cache eviction, numerical stability, multi-layer depth, and full
//! attention blocks.
//!
//! ## GQA Head Grouping & KV-Cache Basics (tests 1-5)
//!
//! 1. GQA head grouping (32Q/8KV -> 4:1) IBP bounds
//! 2. KV-cache append pattern IBP bounds
//! 3. Sliding window with cached positions IBP bounds
//! 4. Multi-head attention output projection IBP bounds
//! 5. Cross-attention between vision encoder and text decoder IBP bounds
//!
//! ## Phase & Position Patterns (tests 6-9)
//!
//! 6. Prefill vs decode phase bound width comparison IBP
//! 7. GQA with RoPE position encoding IBP bounds
//! 8. KV-cache memory layout (interleaved heads) IBP bounds
//! 9. Causal mask interaction with cache offset IBP bounds
//!
//! ## Scaling & Robustness (tests 10-13)
//!
//! 10. GQA at different group ratios (4:1 vs 8:1) IBP comparison
//! 11. Cross-attention with encoder features (VLM pattern) IBP bounds
//! 12. KV-cache eviction/rotation bounds IBP
//! 13. GQA numerical stability (softmax temperature scaling) IBP bounds
//!
//! ## Depth & Composition (tests 14-15)
//!
//! 14. Multi-layer GQA depth composition (2-layer) IBP + CROWN
//! 15. Full attention block: QKV proj + GQA + output proj IBP + CROWN
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, DIM=32, NUM_Q_HEADS=8, NUM_KV_HEADS=2, HEAD_DIM=4
//!
//! Part of #4010: GQA KV-cache attention compose tests for document VLMs.

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

/// Model hidden dimension.
const DIM: usize = 32;
/// Sequence length for inputs.
const SEQ_LEN: usize = 4;
/// Number of query heads (GQA has more Q heads than KV heads).
const NUM_Q_HEADS: usize = 8;
/// Number of KV heads (shared across Q head groups).
const NUM_KV_HEADS: usize = 2;
/// Head dimension = DIM / NUM_Q_HEADS.
const HEAD_DIM: usize = DIM / NUM_Q_HEADS; // 4
/// KV dimension = NUM_KV_HEADS * HEAD_DIM.
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 8
/// FFN intermediate dimension.
const FFN_DIM: usize = 64;
/// Cache length for KV-cache tests (previously generated tokens).
const CACHE_LEN: usize = 8;
/// Sliding window size.
const WINDOW_SIZE: usize = 4;
/// Encoder sequence length for cross-attention tests.
const ENC_SEQ_LEN: usize = 6;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

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
// 1. GQA head grouping (32Q/8KV equivalent, 4:1 ratio) IBP bounds
// ===========================================================================

fn build_gqa_head_grouping_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_gqa_head_grouping");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    // Q projection: [SEQ, DIM] -> [SEQ, DIM] (full Q heads)
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    // K/V projection: [SEQ, DIM] -> [SEQ, KV_DIM] (fewer KV heads)
    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);

    let q_full = b.add_linear(input, q_w, None, &[SEQ_LEN, DIM]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, KV_DIM]);

    // Project Q down to KV_DIM for attention (simulates GQA head grouping)
    let q_down_w = b.add_input("q_down_weight", &[KV_DIM, DIM]);
    let q_down = b.add_linear(input, q_down_w, None, &[SEQ_LEN, KV_DIM]);

    // Attention at KV_DIM
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q_down,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );

    // Project back up: [SEQ, KV_DIM] -> [SEQ, DIM]
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);
    let out = b.add_linear(attn, out_w, None, &[SEQ_LEN, DIM]);

    // Residual
    let _ = q_full; // Full Q unused in simplified verification path
    let result = b.add_binary_add(input, out, &[SEQ_LEN, DIM]);

    b.build(result).expect("valid GQA head grouping kernel")
}

fn gqa_head_grouping_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let q_down_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                 // x
        TensorParamBinding::ConstantTensor(q_w),      // q_weight
        TensorParamBinding::ConstantTensor(k_w),      // k_weight
        TensorParamBinding::ConstantTensor(v_w),      // v_weight
        TensorParamBinding::ConstantTensor(q_down_w), // q_down_weight
        TensorParamBinding::ConstantTensor(out_w),    // out_weight
    ]
}

#[test]
fn test_gqa_head_grouping_ibp() {
    let def = build_gqa_head_grouping_kernel();
    let bindings = gqa_head_grouping_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GQA head grouping");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GQA head grouping (4:1) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "GQA lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "GQA upper must be finite, got {hi_max}");
}

// ===========================================================================
// 2. KV-cache append pattern IBP bounds
// ===========================================================================

/// KV-cache append: current token queries attend over cached KV + current KV.
/// Models the decode-phase pattern where new KV pairs are appended to the cache.
fn build_kv_cache_append_kernel() -> TensorKernelDef {
    let current_tokens: usize = 1;
    let total_kv = CACHE_LEN + current_tokens; // cached + current

    let mut b = TensorBlockBuilder::new("dpdf_gqa_kv_cache_append");

    // Current query: [1, DIM]
    let q_input = b.add_input("query", &[current_tokens, DIM]);
    // Full KV context (cached + current): [total_kv, DIM]
    let kv_input = b.add_input("kv_context", &[total_kv, DIM]);

    // Q/K/V projections (KV uses fewer heads)
    let q_w = b.add_input("q_weight", &[KV_DIM, DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);

    // Project Q from current token, K/V from full context
    let q = b.add_linear(q_input, q_w, None, &[current_tokens, KV_DIM]);
    let k = b.add_linear(kv_input, k_w, None, &[total_kv, KV_DIM]);
    let v = b.add_linear(kv_input, v_w, None, &[total_kv, KV_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[current_tokens, KV_DIM],
    );

    let out = b.add_linear(attn, out_w, None, &[current_tokens, DIM]);

    b.build(out).expect("valid KV-cache append kernel")
}

fn kv_cache_append_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,              // query
        TensorParamBinding::Variable,              // kv_context
        TensorParamBinding::ConstantTensor(q_w),   // q_weight
        TensorParamBinding::ConstantTensor(k_w),   // k_weight
        TensorParamBinding::ConstantTensor(v_w),   // v_weight
        TensorParamBinding::ConstantTensor(out_w), // out_weight
    ]
}

#[test]
fn test_kv_cache_append_ibp() {
    let def = build_kv_cache_append_kernel();
    let bindings = kv_cache_append_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let current_tokens: usize = 1;
    let total_kv = CACHE_LEN + current_tokens;
    let total_seq = current_tokens + total_kv;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through KV-cache append");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("KV-cache append IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "KV-cache append lower must be finite");
    assert!(hi_max.is_finite(), "KV-cache append upper must be finite");
}

// ===========================================================================
// 3. Sliding window with cached positions IBP bounds
// ===========================================================================

/// Sliding window attention over cached positions: only attend to the most
/// recent WINDOW_SIZE positions in the KV cache. Models Qwen3-VL's sliding
/// window pattern during long-context generation.
fn build_sliding_window_cache_kernel() -> TensorKernelDef {
    let current_tokens: usize = 1;
    // Window restricts effective KV to WINDOW_SIZE positions
    let effective_kv = WINDOW_SIZE;

    let mut b = TensorBlockBuilder::new("dpdf_gqa_sliding_window");

    let q_input = b.add_input("query", &[current_tokens, DIM]);
    let kv_input = b.add_input("kv_window", &[effective_kv, DIM]);

    let q_w = b.add_input("q_weight", &[KV_DIM, DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);

    let q = b.add_linear(q_input, q_w, None, &[current_tokens, KV_DIM]);
    let k = b.add_linear(kv_input, k_w, None, &[effective_kv, KV_DIM]);
    let v = b.add_linear(kv_input, v_w, None, &[effective_kv, KV_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[current_tokens, KV_DIM],
    );

    let out = b.add_linear(attn, out_w, None, &[current_tokens, DIM]);

    b.build(out).expect("valid sliding window cache kernel")
}

fn sliding_window_cache_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(q_w),
        TensorParamBinding::ConstantTensor(k_w),
        TensorParamBinding::ConstantTensor(v_w),
        TensorParamBinding::ConstantTensor(out_w),
    ]
}

#[test]
fn test_sliding_window_cache_ibp() {
    let def = build_sliding_window_cache_kernel();
    let bindings = sliding_window_cache_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let current_tokens: usize = 1;
    let effective_kv = WINDOW_SIZE;
    let total_seq = current_tokens + effective_kv;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through sliding window");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sliding window cache IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "sliding window lower must be finite");
    assert!(hi_max.is_finite(), "sliding window upper must be finite");
}

// ===========================================================================
// 4. Multi-head attention output projection IBP bounds
// ===========================================================================

/// Output projection after GQA: attention output at KV_DIM projected back
/// to full model DIM. Verifies the up-projection preserves bounded output.
fn build_output_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_gqa_output_proj");

    // Attention output: [SEQ_LEN, KV_DIM]
    let attn_out = b.add_input("attn_output", &[SEQ_LEN, KV_DIM]);
    // Output projection weight: [DIM, KV_DIM]
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);

    let projected = b.add_linear(attn_out, out_w, None, &[SEQ_LEN, DIM]);

    b.build(projected).expect("valid output projection kernel")
}

#[test]
fn test_output_projection_ibp() {
    let def = build_output_projection_kernel();
    let out_w = ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(out_w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, KV_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through output projection");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GQA output projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "output proj lower must be finite");
    assert!(hi_max.is_finite(), "output proj upper must be finite");
}

// ===========================================================================
// 5. Cross-attention between vision encoder and text decoder IBP bounds
// ===========================================================================

/// Cross-attention: text decoder queries attend to vision encoder features.
/// This is the core VLM fusion pattern (Qwen3-VL, FireRed-OCR).
fn build_vlm_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_gqa_vlm_cross_attn");

    // Text decoder queries: [SEQ_LEN, DIM]
    let q_input = b.add_input("text_query", &[SEQ_LEN, DIM]);
    // Vision encoder features: [ENC_SEQ_LEN, DIM]
    let kv_input = b.add_input("vision_features", &[ENC_SEQ_LEN, DIM]);

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
            NUM_KV_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, DIM],
        )
        .expect("valid VLM cross-attention");

    b.build(out).expect("valid VLM cross-attention kernel")
}

fn vlm_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // text_query
        TensorParamBinding::Variable,                       // vision_features
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

#[test]
fn test_vlm_cross_attention_ibp() {
    let def = build_vlm_cross_attention_kernel();
    let bindings = vlm_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let total_seq = SEQ_LEN + ENC_SEQ_LEN;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through VLM cross-attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("VLM cross-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "VLM cross-attn lower must be finite");
    assert!(hi_max.is_finite(), "VLM cross-attn upper must be finite");
}

// ===========================================================================
// 6. Prefill vs decode phase bound width comparison IBP
// ===========================================================================

/// Compare bound widths between prefill (full sequence) and decode (single token).
/// Prefill processes the full prompt; decode generates one token at a time.
fn build_prefill_or_decode_kernel(seq_len: usize, kv_len: usize) -> TensorKernelDef {
    let name = format!("dpdf_gqa_phase_q{seq_len}_kv{kv_len}");
    let mut b = TensorBlockBuilder::new(&name);

    let q_input = b.add_input("query", &[seq_len, DIM]);
    let kv_input = b.add_input("kv", &[kv_len, DIM]);

    let q_w = b.add_input("q_weight", &[KV_DIM, DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);

    let q = b.add_linear(q_input, q_w, None, &[seq_len, KV_DIM]);
    let k = b.add_linear(kv_input, k_w, None, &[kv_len, KV_DIM]);
    let v = b.add_linear(kv_input, v_w, None, &[kv_len, KV_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[seq_len, KV_DIM],
    );

    let out = b.add_linear(attn, out_w, None, &[seq_len, DIM]);

    b.build(out).expect("valid prefill/decode kernel")
}

fn phase_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(q_w),
        TensorParamBinding::ConstantTensor(k_w),
        TensorParamBinding::ConstantTensor(v_w),
        TensorParamBinding::ConstantTensor(out_w),
    ]
}

#[test]
fn test_prefill_vs_decode_phase_ibp() {
    let bindings = phase_bindings();

    // Prefill: full sequence Q over full sequence KV
    let prefill_def = build_prefill_or_decode_kernel(SEQ_LEN, SEQ_LEN);
    let prefill_graph = tensor_kernel_to_graph(&prefill_def, &bindings).expect("prefill graph");
    let prefill_total = SEQ_LEN + SEQ_LEN;
    let prefill_input = uniform_bounds(&[prefill_total, DIM], 1.0);
    let prefill_output = prefill_graph
        .propagate_ibp(&prefill_input)
        .expect("prefill IBP");
    assert_bounds_valid(&prefill_output);

    // Decode: single token Q over cached + current KV
    let decode_def = build_prefill_or_decode_kernel(1, CACHE_LEN + 1);
    let decode_graph = tensor_kernel_to_graph(&decode_def, &bindings).expect("decode graph");
    let decode_total = 1 + CACHE_LEN + 1;
    let decode_input = uniform_bounds(&[decode_total, DIM], 1.0);
    let decode_output = decode_graph
        .propagate_ibp(&decode_input)
        .expect("decode IBP");
    assert_bounds_valid(&decode_output);

    let prefill_width = bound_width(&prefill_output);
    let decode_width = bound_width(&decode_output);
    eprintln!(
        "Prefill vs decode: prefill width={prefill_width:.6}, decode width={decode_width:.6}"
    );

    // Both should produce finite bounds
    assert!(prefill_width.is_finite(), "prefill width must be finite");
    assert!(decode_width.is_finite(), "decode width must be finite");
}

// ===========================================================================
// 7. GQA with RoPE position encoding IBP bounds
// ===========================================================================

/// GQA attention with sinusoidal positional encoding added before projection.
/// Models the RoPE pattern used in Qwen3-VL and GLM-OCR decoders.
fn build_gqa_rope_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_gqa_rope");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let pe = b.add_input("pos_enc", &[SEQ_LEN, DIM]);

    // Add positional encoding (simulates RoPE application)
    let x_pe = b.add_binary_add(input, pe, &[SEQ_LEN, DIM]);

    // GQA projections
    let q_w = b.add_input("q_weight", &[KV_DIM, DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);

    let q = b.add_linear(x_pe, q_w, None, &[SEQ_LEN, KV_DIM]);
    let k = b.add_linear(x_pe, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(x_pe, v_w, None, &[SEQ_LEN, KV_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );

    let out = b.add_linear(attn, out_w, None, &[SEQ_LEN, DIM]);
    let result = b.add_binary_add(input, out, &[SEQ_LEN, DIM]);

    b.build(result).expect("valid GQA + RoPE kernel")
}

fn gqa_rope_bindings() -> Vec<TensorParamBinding> {
    let pe = super::common::sinusoidal_pe(SEQ_LEN, DIM);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,              // x
        TensorParamBinding::ConstantTensor(pe),    // pos_enc
        TensorParamBinding::ConstantTensor(q_w),   // q_weight
        TensorParamBinding::ConstantTensor(k_w),   // k_weight
        TensorParamBinding::ConstantTensor(v_w),   // v_weight
        TensorParamBinding::ConstantTensor(out_w), // out_weight
    ]
}

#[test]
fn test_gqa_rope_ibp() {
    let def = build_gqa_rope_kernel();
    let bindings = gqa_rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through GQA + RoPE");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GQA + RoPE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "GQA+RoPE lower must be finite");
    assert!(hi_max.is_finite(), "GQA+RoPE upper must be finite");
}

// ===========================================================================
// 8. KV-cache memory layout (interleaved heads) IBP bounds
// ===========================================================================

/// KV-cache with interleaved head layout: cache stores [NUM_KV_HEADS, CACHE_LEN, HEAD_DIM]
/// flattened as [CACHE_LEN, KV_DIM]. Verifies that GQA attention over this layout
/// produces bounded output.
fn build_kv_cache_layout_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_gqa_kv_layout");

    // Query: single current token
    let q_input = b.add_input("query", &[1, DIM]);
    // KV-cache: [CACHE_LEN, KV_DIM] (interleaved head layout)
    let kv_cache = b.add_input("kv_cache", &[CACHE_LEN, KV_DIM]);

    // Q projects down to KV_DIM
    let q_w = b.add_input("q_weight", &[KV_DIM, DIM]);
    let q = b.add_linear(q_input, q_w, None, &[1, KV_DIM]);

    // K/V are projected from the cache (already at KV_DIM)
    let k_w = b.add_input("k_weight", &[KV_DIM, KV_DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, KV_DIM]);
    let k = b.add_linear(kv_cache, k_w, None, &[CACHE_LEN, KV_DIM]);
    let v = b.add_linear(kv_cache, v_w, None, &[CACHE_LEN, KV_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &[1, KV_DIM]);

    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);
    let out = b.add_linear(attn, out_w, None, &[1, DIM]);

    b.build(out).expect("valid KV-cache layout kernel")
}

fn kv_cache_layout_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, KV_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, KV_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,              // query
        TensorParamBinding::Variable,              // kv_cache
        TensorParamBinding::ConstantTensor(q_w),   // q_weight
        TensorParamBinding::ConstantTensor(k_w),   // k_weight
        TensorParamBinding::ConstantTensor(v_w),   // v_weight
        TensorParamBinding::ConstantTensor(out_w), // out_weight
    ]
}

#[test]
fn test_kv_cache_layout_ibp() {
    let def = build_kv_cache_layout_kernel();
    let bindings = kv_cache_layout_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let total_seq = 1 + CACHE_LEN;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through KV-cache layout");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("KV-cache layout IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "KV-cache layout lower must be finite");
    assert!(hi_max.is_finite(), "KV-cache layout upper must be finite");
}

// ===========================================================================
// 9. Causal mask interaction with cache offset IBP bounds
// ===========================================================================

/// Causal masking with KV-cache: during decode, the causal mask must account
/// for the cache offset. Position i in the current sequence can attend to
/// positions 0..cache_len+i in the KV context.
fn build_causal_cache_offset_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_gqa_causal_cache_offset");

    // Small prefill (2 tokens) with causal mask over full KV (cache + current)
    let prefill_len: usize = 2;
    let total_kv = CACHE_LEN + prefill_len;

    let q_input = b.add_input("query", &[prefill_len, DIM]);
    let kv_input = b.add_input("kv_context", &[total_kv, DIM]);

    let q_w = b.add_input("q_weight", &[KV_DIM, DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);

    let q = b.add_linear(q_input, q_w, None, &[prefill_len, KV_DIM]);
    let k = b.add_linear(kv_input, k_w, None, &[total_kv, KV_DIM]);
    let v = b.add_linear(kv_input, v_w, None, &[total_kv, KV_DIM]);

    // Causal mask: each query position can attend to all cached + up to its own position
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[prefill_len, KV_DIM],
    );

    let out = b.add_linear(attn, out_w, None, &[prefill_len, DIM]);

    b.build(out).expect("valid causal cache offset kernel")
}

fn causal_cache_offset_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(q_w),
        TensorParamBinding::ConstantTensor(k_w),
        TensorParamBinding::ConstantTensor(v_w),
        TensorParamBinding::ConstantTensor(out_w),
    ]
}

#[test]
fn test_causal_cache_offset_ibp() {
    let def = build_causal_cache_offset_kernel();
    let bindings = causal_cache_offset_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let prefill_len: usize = 2;
    let total_kv = CACHE_LEN + prefill_len;
    let total_seq = prefill_len + total_kv;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through causal cache offset");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Causal cache offset IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "causal cache lower must be finite");
    assert!(hi_max.is_finite(), "causal cache upper must be finite");
}

// ===========================================================================
// 10. GQA at different group ratios (4:1 vs 8:1) IBP comparison
// ===========================================================================

/// Build a GQA kernel with configurable KV dimension (controlling the group ratio).
fn build_gqa_ratio_kernel(kv_dim: usize) -> TensorKernelDef {
    let name = format!("dpdf_gqa_ratio_kv{kv_dim}");
    let mut b = TensorBlockBuilder::new(&name);

    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    let q_w = b.add_input("q_weight", &[kv_dim, DIM]);
    let k_w = b.add_input("k_weight", &[kv_dim, DIM]);
    let v_w = b.add_input("v_weight", &[kv_dim, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, kv_dim]);

    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, kv_dim]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, kv_dim]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, kv_dim]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, kv_dim],
    );

    let out = b.add_linear(attn, out_w, None, &[SEQ_LEN, DIM]);
    let result = b.add_binary_add(input, out, &[SEQ_LEN, DIM]);

    b.build(result).expect("valid GQA ratio kernel")
}

fn gqa_ratio_bindings(kv_dim: usize) -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[kv_dim, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[kv_dim, DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[kv_dim, DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[DIM, kv_dim]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(q_w),
        TensorParamBinding::ConstantTensor(k_w),
        TensorParamBinding::ConstantTensor(v_w),
        TensorParamBinding::ConstantTensor(out_w),
    ]
}

#[test]
fn test_gqa_ratio_comparison_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    // 4:1 ratio: NUM_Q_HEADS=8, NUM_KV_HEADS=2 -> KV_DIM=8
    let kv_dim_4to1 = KV_DIM; // 8
    let def_4to1 = build_gqa_ratio_kernel(kv_dim_4to1);
    let bindings_4to1 = gqa_ratio_bindings(kv_dim_4to1);
    let graph_4to1 = tensor_kernel_to_graph(&def_4to1, &bindings_4to1).expect("4:1 graph");
    let output_4to1 = graph_4to1.propagate_ibp(&input).expect("4:1 IBP");
    assert_bounds_valid(&output_4to1);

    // 8:1 ratio: NUM_Q_HEADS=8, NUM_KV_HEADS=1 -> KV_DIM=4
    let kv_dim_8to1 = HEAD_DIM; // 4
    let def_8to1 = build_gqa_ratio_kernel(kv_dim_8to1);
    let bindings_8to1 = gqa_ratio_bindings(kv_dim_8to1);
    let graph_8to1 = tensor_kernel_to_graph(&def_8to1, &bindings_8to1).expect("8:1 graph");
    let output_8to1 = graph_8to1.propagate_ibp(&input).expect("8:1 IBP");
    assert_bounds_valid(&output_8to1);

    let width_4to1 = bound_width(&output_4to1);
    let width_8to1 = bound_width(&output_8to1);
    eprintln!("GQA ratio comparison: 4:1 width={width_4to1:.6}, 8:1 width={width_8to1:.6}");

    assert!(width_4to1.is_finite(), "4:1 width must be finite");
    assert!(width_8to1.is_finite(), "8:1 width must be finite");
}

// ===========================================================================
// 11. Cross-attention with encoder features (VLM pattern) IBP bounds
// ===========================================================================

/// Cross-attention with encoder features using GQA: fewer KV heads attend to
/// vision encoder outputs. This tests the VLM cross-attention path where the
/// decoder uses GQA and the encoder features are projected to KV_DIM.
fn build_encoder_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_gqa_encoder_cross_attn");

    // Decoder query: [SEQ_LEN, DIM]
    let q_input = b.add_input("decoder_query", &[SEQ_LEN, DIM]);
    // Encoder features: [ENC_SEQ_LEN, DIM]
    let enc_input = b.add_input("encoder_features", &[ENC_SEQ_LEN, DIM]);

    // Q projects to KV_DIM (GQA), K/V project encoder features to KV_DIM
    let q_w = b.add_input("q_weight", &[KV_DIM, DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);

    let q = b.add_linear(q_input, q_w, None, &[SEQ_LEN, KV_DIM]);
    let k = b.add_linear(enc_input, k_w, None, &[ENC_SEQ_LEN, KV_DIM]);
    let v = b.add_linear(enc_input, v_w, None, &[ENC_SEQ_LEN, KV_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );

    let out = b.add_linear(attn, out_w, None, &[SEQ_LEN, DIM]);
    let result = b.add_binary_add(q_input, out, &[SEQ_LEN, DIM]);

    b.build(result)
        .expect("valid encoder cross-attention kernel")
}

fn encoder_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable, // decoder_query
        TensorParamBinding::Variable, // encoder_features
        TensorParamBinding::ConstantTensor(q_w),
        TensorParamBinding::ConstantTensor(k_w),
        TensorParamBinding::ConstantTensor(v_w),
        TensorParamBinding::ConstantTensor(out_w),
    ]
}

#[test]
fn test_encoder_cross_attention_ibp() {
    let def = build_encoder_cross_attention_kernel();
    let bindings = encoder_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let total_seq = SEQ_LEN + ENC_SEQ_LEN;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder cross-attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Encoder cross-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "encoder cross-attn lower must be finite"
    );
    assert!(
        hi_max.is_finite(),
        "encoder cross-attn upper must be finite"
    );
}

// ===========================================================================
// 12. KV-cache eviction/rotation bounds IBP
// ===========================================================================

/// KV-cache eviction: when cache exceeds max length, oldest entries are evicted.
/// We model this by comparing attention over a short cache (post-eviction) vs
/// a long cache (pre-eviction) to verify bounds remain valid after eviction.
fn build_kv_cache_eviction_kernel(cache_size: usize) -> TensorKernelDef {
    let name = format!("dpdf_gqa_eviction_cache{cache_size}");
    let mut b = TensorBlockBuilder::new(&name);

    let q_input = b.add_input("query", &[1, DIM]);
    let kv_input = b.add_input("kv_cache", &[cache_size, DIM]);

    let q_w = b.add_input("q_weight", &[KV_DIM, DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);

    let q = b.add_linear(q_input, q_w, None, &[1, KV_DIM]);
    let k = b.add_linear(kv_input, k_w, None, &[cache_size, KV_DIM]);
    let v = b.add_linear(kv_input, v_w, None, &[cache_size, KV_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &[1, KV_DIM]);

    let out = b.add_linear(attn, out_w, None, &[1, DIM]);

    b.build(out).expect("valid KV-cache eviction kernel")
}

fn eviction_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(q_w),
        TensorParamBinding::ConstantTensor(k_w),
        TensorParamBinding::ConstantTensor(v_w),
        TensorParamBinding::ConstantTensor(out_w),
    ]
}

#[test]
fn test_kv_cache_eviction_ibp() {
    let bindings = eviction_bindings();

    // Pre-eviction: full cache
    let def_full = build_kv_cache_eviction_kernel(CACHE_LEN);
    let graph_full = tensor_kernel_to_graph(&def_full, &bindings).expect("full cache graph");
    let full_total = 1 + CACHE_LEN;
    let input_full = uniform_bounds(&[full_total, DIM], 1.0);
    let output_full = graph_full
        .propagate_ibp(&input_full)
        .expect("full cache IBP");
    assert_bounds_valid(&output_full);

    // Post-eviction: reduced cache (half size)
    let evicted_len = CACHE_LEN / 2;
    let def_evicted = build_kv_cache_eviction_kernel(evicted_len);
    let graph_evicted = tensor_kernel_to_graph(&def_evicted, &bindings).expect("evicted graph");
    let evicted_total = 1 + evicted_len;
    let input_evicted = uniform_bounds(&[evicted_total, DIM], 1.0);
    let output_evicted = graph_evicted
        .propagate_ibp(&input_evicted)
        .expect("evicted IBP");
    assert_bounds_valid(&output_evicted);

    let full_width = bound_width(&output_full);
    let evicted_width = bound_width(&output_evicted);
    eprintln!("KV-cache eviction: full width={full_width:.6}, evicted width={evicted_width:.6}");

    assert!(full_width.is_finite(), "full cache width must be finite");
    assert!(
        evicted_width.is_finite(),
        "evicted cache width must be finite"
    );
}

// ===========================================================================
// 13. GQA numerical stability (softmax temperature scaling) IBP bounds
// ===========================================================================

/// Test softmax temperature scaling in GQA: using 1/sqrt(d_k) vs no scaling.
/// Proper scaling prevents softmax from saturating with large attention scores.
fn build_gqa_temperature_kernel(use_scale: bool) -> TensorKernelDef {
    let name = if use_scale {
        "dpdf_gqa_temp_scaled"
    } else {
        "dpdf_gqa_temp_unscaled"
    };
    let mut b = TensorBlockBuilder::new(name);

    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    let q_w = b.add_input("q_weight", &[KV_DIM, DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);

    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, KV_DIM]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, KV_DIM]);

    let scale = if use_scale {
        Some(1.0 / (HEAD_DIM as f32).sqrt())
    } else {
        None
    };

    let attn = b.add_attention(q, k, v, AttentionMask::Standard, scale, &[SEQ_LEN, KV_DIM]);

    b.build(attn).expect("valid GQA temperature kernel")
}

fn gqa_temperature_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(q_w),
        TensorParamBinding::ConstantTensor(k_w),
        TensorParamBinding::ConstantTensor(v_w),
    ]
}

#[test]
fn test_gqa_numerical_stability_ibp() {
    let bindings = gqa_temperature_bindings();
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let def_scaled = build_gqa_temperature_kernel(true);
    let graph_scaled = tensor_kernel_to_graph(&def_scaled, &bindings).expect("scaled graph");
    let output_scaled = graph_scaled.propagate_ibp(&input).expect("scaled IBP");
    assert_bounds_valid(&output_scaled);

    let def_unscaled = build_gqa_temperature_kernel(false);
    let graph_unscaled = tensor_kernel_to_graph(&def_unscaled, &bindings).expect("unscaled graph");
    let output_unscaled = graph_unscaled.propagate_ibp(&input).expect("unscaled IBP");
    assert_bounds_valid(&output_unscaled);

    let scaled_width = bound_width(&output_scaled);
    let unscaled_width = bound_width(&output_unscaled);
    eprintln!(
        "GQA temperature: scaled width={scaled_width:.6}, unscaled width={unscaled_width:.6}"
    );

    assert!(scaled_width.is_finite(), "scaled width must be finite");
    assert!(unscaled_width.is_finite(), "unscaled width must be finite");
}

// ===========================================================================
// 14. Multi-layer GQA depth composition (2-layer) IBP + CROWN
// ===========================================================================

/// Build a 2-layer GQA decoder: each layer has RMSNorm -> GQA -> residual -> RMSNorm -> SwiGLU -> residual.
fn build_two_layer_gqa_decoder_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let mut b = TensorBlockBuilder::new("dpdf_gqa_two_layer_decoder");

    let input = b.add_input("x", &shape);

    // Helper closure-like macro for a single decoder layer
    let mut current = input;
    for layer_idx in 0..2 {
        let prefix = format!("l{layer_idx}_");

        // Pre-attention RMSNorm
        let n1_eps = b.add_input(&format!("{prefix}norm1_eps"), &[1]);
        let n1_w = b.add_input(&format!("{prefix}norm1_weight"), &[DIM]);
        let normed1 = b.add_rms_norm(current, n1_eps, 1, n1_w, &shape);

        // GQA attention (using KV_DIM for reduced KV heads)
        let q_w = b.add_input(&format!("{prefix}q_weight"), &[KV_DIM, DIM]);
        let k_w = b.add_input(&format!("{prefix}k_weight"), &[KV_DIM, DIM]);
        let v_w = b.add_input(&format!("{prefix}v_weight"), &[KV_DIM, DIM]);
        let out_w = b.add_input(&format!("{prefix}out_weight"), &[DIM, KV_DIM]);

        let q = b.add_linear(normed1, q_w, None, &[SEQ_LEN, KV_DIM]);
        let k = b.add_linear(normed1, k_w, None, &[SEQ_LEN, KV_DIM]);
        let v = b.add_linear(normed1, v_w, None, &[SEQ_LEN, KV_DIM]);

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let attn = b.add_attention(
            q,
            k,
            v,
            AttentionMask::Causal,
            Some(scale),
            &[SEQ_LEN, KV_DIM],
        );
        let attn_out = b.add_linear(attn, out_w, None, &shape);

        // Residual after attention
        let res1 = b.add_binary_add(current, attn_out, &shape);

        // Pre-FFN RMSNorm
        let n2_eps = b.add_input(&format!("{prefix}norm2_eps"), &[1]);
        let n2_w = b.add_input(&format!("{prefix}norm2_weight"), &[DIM]);
        let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

        // SwiGLU FFN
        let gate_w = b.add_input(&format!("{prefix}gate_weight"), &[FFN_DIM, DIM]);
        let up_w = b.add_input(&format!("{prefix}up_weight"), &[FFN_DIM, DIM]);
        let down_w = b.add_input(&format!("{prefix}down_weight"), &[DIM, FFN_DIM]);

        let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
        let gate_act = add_silu(&mut b, gate, &ffn_shape);
        let up = b.add_linear(normed2, up_w, None, &ffn_shape);
        let gated = b.add_binary_mul(gate_act, up, &ffn_shape);
        let down = b.add_linear(gated, down_w, None, &shape);

        // Residual after FFN
        current = b.add_binary_add(res1, down, &shape);
    }

    b.build(current).expect("valid 2-layer GQA decoder kernel")
}

fn two_layer_gqa_decoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // x

    for _ in 0..2 {
        // RMSNorm 1
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DIM]),
            1.0f32,
        ))); // norm1_weight

        // GQA attention weights
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[KV_DIM, DIM]),
            WEIGHT_MAG,
        ))); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[KV_DIM, DIM]),
            WEIGHT_MAG,
        ))); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[KV_DIM, DIM]),
            WEIGHT_MAG,
        ))); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DIM, KV_DIM]),
            WEIGHT_MAG,
        ))); // out_weight

        // RMSNorm 2
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DIM]),
            1.0f32,
        ))); // norm2_weight

        // SwiGLU FFN weights
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, DIM]),
            WEIGHT_MAG,
        ))); // gate_weight
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, DIM]),
            WEIGHT_MAG,
        ))); // up_weight
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DIM, FFN_DIM]),
            WEIGHT_MAG,
        ))); // down_weight
    }

    bindings
}

#[test]
fn test_two_layer_gqa_decoder_ibp() {
    let def = build_two_layer_gqa_decoder_kernel();
    let bindings = two_layer_gqa_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-layer GQA decoder");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("2-layer GQA decoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "2-layer GQA lower must be finite");
    assert!(hi_max.is_finite(), "2-layer GQA upper must be finite");
}

#[test]
fn test_two_layer_gqa_decoder_crown() {
    let def = build_two_layer_gqa_decoder_kernel();
    let bindings = two_layer_gqa_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("2-layer GQA decoder CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 15. Full attention block: QKV proj + GQA + output proj IBP + CROWN
// ===========================================================================

/// Full GQA attention block with RMSNorm, QKV projection, attention, output
/// projection, and residual. This is the complete attention sub-layer as used
/// in Qwen3-VL and GLM-OCR decoder layers.
fn build_full_gqa_attention_block_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, DIM];
    let mut b = TensorBlockBuilder::new("dpdf_gqa_full_attn_block");

    let input = b.add_input("x", &shape);

    // Pre-attention RMSNorm
    let n_eps = b.add_input("norm_eps", &[1]);
    let n_w = b.add_input("norm_weight", &[DIM]);
    let normed = b.add_rms_norm(input, n_eps, 1, n_w, &shape);

    // GQA Q/K/V projections
    let q_w = b.add_input("q_weight", &[DIM, DIM]); // Full Q heads
    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]); // Reduced KV heads
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);

    let q_full = b.add_linear(normed, q_w, None, &[SEQ_LEN, DIM]);

    // Project Q down to KV_DIM for attention
    let q_down_w = b.add_input("q_down_weight", &[KV_DIM, DIM]);
    let q_down = b.add_linear(normed, q_down_w, None, &[SEQ_LEN, KV_DIM]);

    let k = b.add_linear(normed, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(normed, v_w, None, &[SEQ_LEN, KV_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q_down,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );

    // Output projection: [SEQ, KV_DIM] -> [SEQ, DIM]
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);
    let out = b.add_linear(attn, out_w, None, &shape);

    // Residual
    let _ = q_full; // Full Q unused in simplified verification path
    let result = b.add_binary_add(input, out, &shape);

    b.build(result)
        .expect("valid full GQA attention block kernel")
}

fn full_gqa_attention_block_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // norm_eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 1.0f32)), // norm_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)), // q_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG)), // k_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG)), // v_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG)), // q_down_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG)), // out_weight
    ]
}

#[test]
fn test_full_gqa_attention_block_ibp() {
    let def = build_full_gqa_attention_block_kernel();
    let bindings = full_gqa_attention_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full GQA attention block");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full GQA attention block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "full GQA block lower must be finite");
    assert!(hi_max.is_finite(), "full GQA block upper must be finite");
}

#[test]
fn test_full_gqa_attention_block_crown() {
    let def = build_full_gqa_attention_block_kernel();
    let bindings = full_gqa_attention_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "Full GQA attention block CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}
