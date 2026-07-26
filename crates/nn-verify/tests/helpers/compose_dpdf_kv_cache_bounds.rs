// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for KV cache bounds growth and memory safety.
//!
//! Verifies IBP and CROWN bound propagation through KV cache patterns critical
//! to autoregressive LLM inference in dpdf document understanding models
//! (Qwen3-VL, GLM-OCR, Granite-Docling). KV caches store projected key/value
//! tensors from previous decoding steps; bounds must remain tight as caches
//! grow, entries are concatenated, and position encodings shift.
//!
//! ## Cache Shape & Growth (tests 1-4)
//!
//! 1.  **Cache shape after 1 step** (IBP)
//! 2.  **Cache shape after 4 steps** (IBP)
//! 3.  **Cache shape after 8 steps (saturation)** (IBP)
//! 4.  **Bounds bounded by activation range** (IBP)
//!
//! ## Concatenation Preservation (tests 5-8)
//!
//! 5.  **Concatenation preserves existing entries** (IBP)
//! 6.  **Concatenation append order invariance** (IBP)
//! 7.  **Multi-step concatenation chain** (IBP + CROWN)
//! 8.  **Concatenation with projection preserves bounds** (IBP)
//!
//! ## Paged KV & GQA (tests 9-12)
//!
//! 9.  **Paged KV block allocation (fixed block size)** (IBP)
//! 10. **Paged KV cross-block attention** (IBP)
//! 11. **GQA cache sharing (fewer KV heads)** (IBP)
//! 12. **GQA cache sharing with attention output** (IBP + CROWN)
//!
//! ## RoPE & Sliding Window (tests 13-18)
//!
//! 13. **RoPE with cache offset (position shift)** (IBP)
//! 14. **RoPE offset monotone bound growth** (IBP)
//! 15. **Sliding window cache eviction bounds** (IBP)
//! 16. **Sliding window cache with attention** (IBP)
//! 17. **Full autoregressive step with KV cache** (IBP + CROWN)
//! 18. **Multi-layer KV cache depth composition** (IBP + CROWN)
//!
//! Architecture references:
//! - KV cache (Vaswani et al., 2017): Store K/V projections for O(1) per-step decode
//! - Grouped-Query Attention (Ainslie et al., 2023): Fewer KV heads shared across Q groups
//! - RoPE (Su et al., 2021): Rotary positional encoding with absolute position offset
//! - PagedAttention (Kwon et al., 2023): Fixed-size KV blocks for memory efficiency
//! - Sliding window attention (Beltagy et al., 2020): Bounded cache for long contexts
//!
//! Dimensions (small for fast verification, structurally representative):
//! - DIM=16, NUM_HEADS=4, HEAD_DIM=4, NUM_KV_HEADS=2, KV_DIM=8
//! - CACHE_STEP increments of 1 token, max CACHE_LEN=8
//! - WINDOW_SIZE=4, PAGE_SIZE=2
//!
//! Part of #4136: Compose tests for KV cache bounds growth and memory safety.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Model hidden dimension.
const DIM: usize = 16;
/// Number of query heads.
const NUM_HEADS: usize = 4;
/// Per-head dimension.
const HEAD_DIM: usize = DIM / NUM_HEADS; // 4
/// Number of KV heads (GQA: fewer than Q heads).
const NUM_KV_HEADS: usize = 2;
/// KV cache dimension = NUM_KV_HEADS * HEAD_DIM.
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 8
/// Maximum cache length (number of previously decoded tokens).
const CACHE_LEN: usize = 8;
/// Sliding window size for bounded-context tests.
const WINDOW_SIZE: usize = 4;
/// Page size for paged KV block tests.
const PAGE_SIZE: usize = 2;
/// FFN intermediate dimension.
const FFN_DIM: usize = 32;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Zero bias tensor binding.
fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Ones tensor binding (for RMSNorm / LayerNorm weight).
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Scalar epsilon binding.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
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

/// Build a K/V projection kernel: input [seq, DIM] -> projected [seq, KV_DIM].
///
/// Returns (kernel_def, bindings).
fn build_kv_projection_kernel(
    name: &str,
    seq_len: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("x", &[seq_len, DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, DIM]);
    let k = b.add_linear(input, k_w, None, &[seq_len, KV_DIM]);
    let v = b.add_linear(input, v_w, None, &[seq_len, KV_DIM]);
    // Output is concatenated K and V along feature dim: [seq, 2*KV_DIM]
    let out = b.add_concat(&[k, v], 1, &[seq_len, 2 * KV_DIM]);
    let def = b.build(out).expect("valid KV projection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[KV_DIM, DIM]),
        weight(&[KV_DIM, DIM]),
    ];
    (def, bindings)
}

/// Build a cache-attended attention kernel.
///
/// Q: [1, KV_DIM], K: [cache_len, KV_DIM], V: [cache_len, KV_DIM]
/// Output: [1, KV_DIM] after attention.
fn build_cache_attention_kernel(
    name: &str,
    cache_len: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new(name);
    let q = b.add_input("query", &[1, KV_DIM]);
    let k = b.add_input("cached_k", &[cache_len, KV_DIM]);
    let v = b.add_input("cached_v", &[cache_len, KV_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &[1, KV_DIM]);
    let def = b.build(attn).expect("valid cache attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // query
        TensorParamBinding::Variable, // cached_k
        TensorParamBinding::Variable, // cached_v
    ];
    (def, bindings)
}

// ===========================================================================
// 1. Cache shape after 1 step (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_shape_after_1_step_ibp() {
    // After 1 decode step: cache holds 1 token's K/V projections.
    let (def, bindings) = build_kv_projection_kernel("kv_cache_step1", 1);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[1, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, 2 * KV_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("KV cache step=1 IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. Cache shape after 4 steps (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_shape_after_4_steps_ibp() {
    // After 4 decode steps: cache holds 4 tokens' K/V projections.
    let (def, bindings) = build_kv_projection_kernel("kv_cache_step4", 4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[4, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[4, 2 * KV_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("KV cache step=4 IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. Cache shape after 8 steps (saturation) (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_shape_after_8_steps_ibp() {
    // At CACHE_LEN=8, the cache is full. Verify shape and bounds.
    let (def, bindings) = build_kv_projection_kernel("kv_cache_step8", CACHE_LEN);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CACHE_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[CACHE_LEN, 2 * KV_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("KV cache step=8 (full) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 4. Bounds bounded by activation range (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_bounds_within_activation_range_ibp() {
    // K/V projections are linear transforms of bounded input.
    // Output bounds should be proportional to input_range * weight_mag * DIM.
    let (def, bindings) = build_kv_projection_kernel("kv_cache_activation_range", 4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_range = 1.0;
    let input = uniform_bounds(&[4, DIM], input_range);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("KV cache activation range IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // Analytical bound: linear output = W @ x, max |output| <= DIM * WEIGHT_MAG * input_range
    let analytical_max = (DIM as f32) * WEIGHT_MAG * input_range;
    assert!(
        hi_max <= analytical_max + 1e-4,
        "upper bound {hi_max} should be <= analytical {analytical_max}"
    );
    assert!(
        lo_min >= -analytical_max - 1e-4,
        "lower bound {lo_min} should be >= analytical {analytical_max}"
    );
}

// ===========================================================================
// 5. Concatenation preserves existing entries (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_concat_preserves_entries_ibp() {
    // Concatenate cached KV [4, KV_DIM] with new token KV [1, KV_DIM].
    // Verify the concatenated output [5, KV_DIM] preserves bounds from both.
    let cached_len: usize = 4;
    let new_len: usize = 1;
    let total = cached_len + new_len;

    let mut b = TensorBlockBuilder::new("kv_cache_concat_preserve");
    let cached = b.add_input("cached_kv", &[cached_len, KV_DIM]);
    let new_kv = b.add_input("new_kv", &[new_len, KV_DIM]);
    let out = b.add_concat(&[cached, new_kv], 0, &[total, KV_DIM]);
    let def = b.build(out).expect("valid concat kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // cached_kv
        TensorParamBinding::Variable, // new_kv
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Multi-variable: [2, total_elements] flattened
    let input = uniform_bounds(&[2, total, KV_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[total, KV_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("KV concat preserve IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Concat should not widen beyond input range
    assert!(hi_max <= 1.0 + 1e-4, "concat should not widen upper bound");
    assert!(lo_min >= -1.0 - 1e-4, "concat should not widen lower bound");
}

// ===========================================================================
// 6. Concatenation append order invariance (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_concat_order_invariance_ibp() {
    // Two separate concat patterns produce same shape. Both should have
    // valid bounds (order should not break verification).
    let part_a: usize = 3;
    let part_b: usize = 3;
    let total = part_a + part_b;

    // Pattern 1: concat(A, B)
    let mut b1 = TensorBlockBuilder::new("kv_cache_concat_order1");
    let a1 = b1.add_input("part_a", &[part_a, KV_DIM]);
    let b1_node = b1.add_input("part_b", &[part_b, KV_DIM]);
    let out1 = b1.add_concat(&[a1, b1_node], 0, &[total, KV_DIM]);
    let def1 = b1.build(out1).expect("valid concat order1 kernel");

    // Pattern 2: concat(B, A)
    let mut b2 = TensorBlockBuilder::new("kv_cache_concat_order2");
    let a2 = b2.add_input("part_a", &[part_a, KV_DIM]);
    let b2_node = b2.add_input("part_b", &[part_b, KV_DIM]);
    let out2 = b2.add_concat(&[b2_node, a2], 0, &[total, KV_DIM]);
    let def2 = b2.build(out2).expect("valid concat order2 kernel");

    let bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];
    let graph1 = tensor_kernel_to_graph(&def1, &bindings).expect("graph1");
    let graph2 = tensor_kernel_to_graph(&def2, &bindings).expect("graph2");
    let input = uniform_bounds(&[2, total, KV_DIM], 1.0);

    let out_1 = graph1.propagate_ibp(&input).expect("IBP order1");
    let out_2 = graph2.propagate_ibp(&input).expect("IBP order2");
    assert_bounds_valid(&out_1);
    assert_bounds_valid(&out_2);

    let (lo1, hi1) = bounds_min_max(&out_1);
    let (lo2, hi2) = bounds_min_max(&out_2);
    eprintln!("Concat order1 IBP: [{lo1:.6}, {hi1:.6}]");
    eprintln!("Concat order2 IBP: [{lo2:.6}, {hi2:.6}]");
    // Both should be finite with same input range
    assert!(lo1.is_finite() && hi1.is_finite());
    assert!(lo2.is_finite() && hi2.is_finite());
}

// ===========================================================================
// 7. Multi-step concatenation chain (IBP + CROWN)
// ===========================================================================

#[test]
fn test_kv_cache_multi_step_concat_ibp_crown() {
    // Chain: step0 [2, KV_DIM] -> concat step1 [1, KV_DIM] -> [3, KV_DIM]
    //        -> concat step2 [1, KV_DIM] -> [4, KV_DIM] -> attention
    let step0: usize = 2;
    let step1: usize = 1;
    let step2: usize = 1;
    let total = step0 + step1 + step2; // 4

    let mut b = TensorBlockBuilder::new("kv_cache_multi_step_concat");
    let cache0 = b.add_input("cache_step0", &[step0, KV_DIM]);
    let new1 = b.add_input("new_step1", &[step1, KV_DIM]);
    let cache1 = b.add_concat(&[cache0, new1], 0, &[step0 + step1, KV_DIM]);
    let new2 = b.add_input("new_step2", &[step2, KV_DIM]);
    let cache2 = b.add_concat(&[cache1, new2], 0, &[total, KV_DIM]);

    // Single-token query attends to full cache
    let query = b.add_input("query", &[1, KV_DIM]);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        query,
        cache2,
        cache2,
        AttentionMask::Standard,
        Some(scale),
        &[1, KV_DIM],
    );
    let def = b.build(attn).expect("valid multi-step concat kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // cache_step0
        TensorParamBinding::Variable, // new_step1
        TensorParamBinding::Variable, // new_step2
        TensorParamBinding::Variable, // query
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // 4 variable inputs: [4, ...]
    let input = uniform_bounds(&[4, total, KV_DIM], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Multi-step concat IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Multi-step concat CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 8. Concatenation with projection preserves bounds (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_concat_with_projection_ibp() {
    // Concat cached + new, then project through linear layer.
    // Verifies that linear projection after concat preserves bound validity.
    let cached_len: usize = 4;
    let new_len: usize = 1;
    let total = cached_len + new_len;

    let mut b = TensorBlockBuilder::new("kv_cache_concat_proj");
    let cached = b.add_input("cached_kv", &[cached_len, KV_DIM]);
    let new_kv = b.add_input("new_kv", &[new_len, KV_DIM]);
    let concat = b.add_concat(&[cached, new_kv], 0, &[total, KV_DIM]);

    // Project to DIM
    let proj_w = b.add_input("proj_w", &[DIM, KV_DIM]);
    let out = b.add_linear(concat, proj_w, None, &[total, DIM]);
    let def = b.build(out).expect("valid concat+proj kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        weight(&[DIM, KV_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[2, total, KV_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[total, DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Concat+proj IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 9. Paged KV block allocation (fixed block size) (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_paged_block_allocation_ibp() {
    // Paged KV: cache is organized as fixed-size blocks of PAGE_SIZE tokens.
    // Simulate: 2 blocks of [PAGE_SIZE, KV_DIM] concatenated -> [2*PAGE_SIZE, KV_DIM].
    let num_blocks: usize = 2;
    let total = num_blocks * PAGE_SIZE;

    let mut b = TensorBlockBuilder::new("kv_cache_paged_blocks");
    let block0 = b.add_input("block0", &[PAGE_SIZE, KV_DIM]);
    let block1 = b.add_input("block1", &[PAGE_SIZE, KV_DIM]);
    let cache = b.add_concat(&[block0, block1], 0, &[total, KV_DIM]);

    // Query attends to paged cache
    let query = b.add_input("query", &[1, KV_DIM]);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        query,
        cache,
        cache,
        AttentionMask::Standard,
        Some(scale),
        &[1, KV_DIM],
    );
    let def = b.build(attn).expect("valid paged KV kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // block0
        TensorParamBinding::Variable, // block1
        TensorParamBinding::Variable, // query
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[3, total, KV_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Paged KV ({num_blocks} blocks, page={PAGE_SIZE}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. Paged KV cross-block attention (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_paged_cross_block_attention_ibp() {
    // 4 pages concatenated: query must attend across all pages.
    let num_blocks: usize = 4;
    let total = num_blocks * PAGE_SIZE;

    let mut b = TensorBlockBuilder::new("kv_cache_paged_cross_block");
    let blocks: Vec<_> = (0..num_blocks)
        .map(|i| b.add_input(&format!("block{i}"), &[PAGE_SIZE, KV_DIM]))
        .collect();
    let cache = b.add_concat(&blocks, 0, &[total, KV_DIM]);

    let query = b.add_input("query", &[1, KV_DIM]);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        query,
        cache,
        cache,
        AttentionMask::Standard,
        Some(scale),
        &[1, KV_DIM],
    );
    let def = b.build(attn).expect("valid paged cross-block kernel");

    let mut bindings: Vec<TensorParamBinding> = (0..num_blocks)
        .map(|_| TensorParamBinding::Variable)
        .collect();
    bindings.push(TensorParamBinding::Variable); // query

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let num_vars = num_blocks + 1;
    let input = uniform_bounds(&[num_vars, total, KV_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paged cross-block ({num_blocks} pages) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 11. GQA cache sharing (fewer KV heads) (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_gqa_sharing_ibp() {
    // GQA: Q has NUM_HEADS heads, KV has NUM_KV_HEADS heads.
    // Q is projected to DIM, KV to KV_DIM. Q is down-projected to KV_DIM
    // to match KV for attention (simulates head group sharing).
    let cache_len: usize = 4;

    let mut b = TensorBlockBuilder::new("kv_cache_gqa_sharing");
    let input = b.add_input("x", &[1, DIM]);
    let cached_k = b.add_input("cached_k", &[cache_len, KV_DIM]);
    let cached_v = b.add_input("cached_v", &[cache_len, KV_DIM]);

    // Q projection: [1, DIM] -> [1, KV_DIM] (down-projected for GQA)
    let q_w = b.add_input("q_weight", &[KV_DIM, DIM]);
    let q = b.add_linear(input, q_w, None, &[1, KV_DIM]);

    // Attention with cached K/V (GQA sharing)
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        cached_k,
        cached_v,
        AttentionMask::Standard,
        Some(scale),
        &[1, KV_DIM],
    );

    // Project back to full DIM
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);
    let out = b.add_linear(attn, out_w, None, &[1, DIM]);
    let result = b.add_binary_add(input, out, &[1, DIM]);
    let def = b.build(result).expect("valid GQA sharing kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::Variable, // cached_k
        TensorParamBinding::Variable, // cached_v
        weight(&[KV_DIM, DIM]),       // q_weight
        weight(&[DIM, KV_DIM]),       // out_weight
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[3, cache_len, KV_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GQA cache sharing ({NUM_KV_HEADS} KV heads) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 12. GQA cache sharing with attention output (IBP + CROWN)
// ===========================================================================

#[test]
fn test_kv_cache_gqa_sharing_attention_ibp_crown() {
    // GQA attention with output projection. Test both IBP and CROWN.
    let cache_len: usize = 4;

    let mut b = TensorBlockBuilder::new("kv_cache_gqa_attn_output");
    let query = b.add_input("query", &[1, KV_DIM]);
    let cached_k = b.add_input("cached_k", &[cache_len, KV_DIM]);
    let cached_v = b.add_input("cached_v", &[cache_len, KV_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        query,
        cached_k,
        cached_v,
        AttentionMask::Standard,
        Some(scale),
        &[1, KV_DIM],
    );

    // Output projection
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);
    let out = b.add_linear(attn, out_w, None, &[1, DIM]);
    let def = b.build(out).expect("valid GQA attn output kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // query
        TensorParamBinding::Variable, // cached_k
        TensorParamBinding::Variable, // cached_v
        weight(&[DIM, KV_DIM]),       // out_weight
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[3, cache_len, KV_DIM], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GQA attn output IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GQA attn output CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 13. RoPE with cache offset (position shift) (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_rope_with_offset_ibp() {
    // RoPE at decode step: position = cache_offset + step.
    // Simulate RoPE as element-wise modulation: x * cos(pos) + rotate(x) * sin(pos).
    // For verification, model as: x * cos_const + x * sin_const (linear combination).
    // At offset=4 (after 4 cached tokens), cos/sin values differ from offset=0.
    let cache_offset: usize = 4;

    let mut b = TensorBlockBuilder::new("kv_cache_rope_offset");
    let input = b.add_input("x", &[1, KV_DIM]);

    // RoPE cos/sin at position = cache_offset (precomputed constants)
    let cos_vals: Vec<f32> = (0..KV_DIM)
        .map(|i| {
            let freq =
                (cache_offset as f64) / 10000.0_f64.powf(2.0 * (i / 2) as f64 / KV_DIM as f64);
            freq.cos() as f32
        })
        .collect();
    let sin_vals: Vec<f32> = (0..KV_DIM)
        .map(|i| {
            let freq =
                (cache_offset as f64) / 10000.0_f64.powf(2.0 * (i / 2) as f64 / KV_DIM as f64);
            freq.sin() as f32
        })
        .collect();

    let cos_node = b.add_input("cos_pos", &[1, KV_DIM]);
    let sin_node = b.add_input("sin_pos", &[1, KV_DIM]);

    // RoPE: x * cos + x * sin (simplified, treats rotate(x) as x for bound analysis)
    let x_cos = b.add_binary_mul(input, cos_node, &[1, KV_DIM]);
    let x_sin = b.add_binary_mul(input, sin_node, &[1, KV_DIM]);
    let out = b.add_binary_add(x_cos, x_sin, &[1, KV_DIM]);
    let def = b.build(out).expect("valid RoPE offset kernel");

    let cos_tensor = ArrayD::from_shape_vec(IxDyn(&[1, KV_DIM]), cos_vals).unwrap();
    let sin_tensor = ArrayD::from_shape_vec(IxDyn(&[1, KV_DIM]), sin_vals).unwrap();

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_tensor),
        TensorParamBinding::ConstantTensor(sin_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[1, KV_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RoPE offset={cache_offset} IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // RoPE should not explode bounds: |cos| <= 1, |sin| <= 1, so output <= 2 * input
    assert!(
        hi_max <= 2.0 + 1e-4,
        "RoPE should not exceed 2x input range"
    );
    assert!(
        lo_min >= -2.0 - 1e-4,
        "RoPE should not exceed 2x input range"
    );
}

// ===========================================================================
// 14. RoPE offset monotone bound growth (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_rope_offset_monotone_ibp() {
    // As cache_offset increases, RoPE bounds should remain stable (bounded by [-2, 2]
    // for unit input range), verifying that position shift does not cause blow-up.
    let offsets = [0_usize, 4, 8, 16];
    let mut prev_width: Option<f32> = None;

    for &offset in &offsets {
        let mut b = TensorBlockBuilder::new(&format!("kv_cache_rope_offset_{offset}"));
        let input = b.add_input("x", &[1, KV_DIM]);
        let cos_node = b.add_input("cos_pos", &[1, KV_DIM]);
        let sin_node = b.add_input("sin_pos", &[1, KV_DIM]);
        let x_cos = b.add_binary_mul(input, cos_node, &[1, KV_DIM]);
        let x_sin = b.add_binary_mul(input, sin_node, &[1, KV_DIM]);
        let out = b.add_binary_add(x_cos, x_sin, &[1, KV_DIM]);
        let def = b.build(out).expect("valid RoPE offset kernel");

        let cos_vals: Vec<f32> = (0..KV_DIM)
            .map(|i| {
                let freq = (offset as f64) / 10000.0_f64.powf(2.0 * (i / 2) as f64 / KV_DIM as f64);
                freq.cos() as f32
            })
            .collect();
        let sin_vals: Vec<f32> = (0..KV_DIM)
            .map(|i| {
                let freq = (offset as f64) / 10000.0_f64.powf(2.0 * (i / 2) as f64 / KV_DIM as f64);
                freq.sin() as f32
            })
            .collect();

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(
                ArrayD::from_shape_vec(IxDyn(&[1, KV_DIM]), cos_vals).unwrap(),
            ),
            TensorParamBinding::ConstantTensor(
                ArrayD::from_shape_vec(IxDyn(&[1, KV_DIM]), sin_vals).unwrap(),
            ),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let inp = uniform_bounds(&[1, KV_DIM], 1.0);
        let output = graph.propagate_ibp(&inp).expect("IBP");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("RoPE offset={offset}: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");

        // Width should stay bounded (not grow unboundedly with offset)
        assert!(
            width <= 4.1,
            "RoPE width at offset={offset} should be <= 4.1, got {width}"
        );

        if let Some(pw) = prev_width {
            // Width should not grow dramatically
            assert!(
                width <= pw * 2.0 + 1e-3,
                "RoPE width at offset={offset} ({width}) should not be > 2x previous ({pw})"
            );
        }
        prev_width = Some(width);
    }
}

// ===========================================================================
// 15. Sliding window cache eviction bounds (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_sliding_window_eviction_ibp() {
    // Sliding window: only keep the most recent WINDOW_SIZE tokens in cache.
    // After 8 steps with WINDOW_SIZE=4, the cache holds tokens [4..8].
    // Model as: full [CACHE_LEN, KV_DIM] -> narrow to [WINDOW_SIZE, KV_DIM].
    let mut b = TensorBlockBuilder::new("kv_cache_sliding_window_evict");
    let full_cache = b.add_input("full_cache", &[CACHE_LEN, KV_DIM]);

    // Evict oldest: narrow to last WINDOW_SIZE entries
    let eviction_start = CACHE_LEN - WINDOW_SIZE;
    let windowed = b.add_narrow(
        full_cache,
        0,
        eviction_start,
        WINDOW_SIZE,
        &[WINDOW_SIZE, KV_DIM],
    );
    let def = b.build(windowed).expect("valid sliding window kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CACHE_LEN, KV_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[WINDOW_SIZE, KV_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sliding window eviction IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Narrow should not widen bounds beyond input range
    assert!(hi_max <= 1.0 + 1e-4, "narrow should not widen bounds");
    assert!(lo_min >= -1.0 - 1e-4, "narrow should not widen bounds");
}

// ===========================================================================
// 16. Sliding window cache with attention (IBP)
// ===========================================================================

#[test]
fn test_kv_cache_sliding_window_attention_ibp() {
    // Sliding window: narrow cache to WINDOW_SIZE, then attend.
    let mut b = TensorBlockBuilder::new("kv_cache_sliding_window_attn");
    let full_cache = b.add_input("full_cache", &[CACHE_LEN, KV_DIM]);
    let query = b.add_input("query", &[1, KV_DIM]);

    // Narrow to window
    let eviction_start = CACHE_LEN - WINDOW_SIZE;
    let windowed_k = b.add_narrow(
        full_cache,
        0,
        eviction_start,
        WINDOW_SIZE,
        &[WINDOW_SIZE, KV_DIM],
    );
    let windowed_v = b.add_narrow(
        full_cache,
        0,
        eviction_start,
        WINDOW_SIZE,
        &[WINDOW_SIZE, KV_DIM],
    );

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        query,
        windowed_k,
        windowed_v,
        AttentionMask::Standard,
        Some(scale),
        &[1, KV_DIM],
    );
    let def = b
        .build(attn)
        .expect("valid sliding window attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // full_cache
        TensorParamBinding::Variable, // query
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[2, CACHE_LEN, KV_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, KV_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sliding window attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 17. Full autoregressive step with KV cache (IBP + CROWN)
// ===========================================================================

#[test]
fn test_kv_cache_full_autoregressive_step_ibp_crown() {
    // Full autoregressive step:
    // 1. RMSNorm(input)
    // 2. Q/K/V projection
    // 3. Attention with cached K/V
    // 4. Residual connection
    // 5. RMSNorm -> SwiGLU FFN -> residual
    let cache_len: usize = 4;

    let mut b = TensorBlockBuilder::new("kv_cache_full_auto_step");
    let input = b.add_input("x", &[1, DIM]);
    let cached_k = b.add_input("cached_k", &[cache_len, KV_DIM]);
    let cached_v = b.add_input("cached_v", &[cache_len, KV_DIM]);

    // RMSNorm
    let rn1_w = b.add_input("rn1_w", &[DIM]);
    let rn1_eps = b.add_input("rn1_eps", &[1]);
    let normed = b.add_rms_norm(input, rn1_eps, 1, rn1_w, &[1, DIM]);

    // Q projection -> KV_DIM
    let q_w = b.add_input("q_weight", &[KV_DIM, DIM]);
    let q = b.add_linear(normed, q_w, None, &[1, KV_DIM]);

    // Attention with cache
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        cached_k,
        cached_v,
        AttentionMask::Standard,
        Some(scale),
        &[1, KV_DIM],
    );

    // Output projection + residual
    let out_w = b.add_input("out_weight", &[DIM, KV_DIM]);
    let attn_out = b.add_linear(attn, out_w, None, &[1, DIM]);
    let res1 = b.add_binary_add(input, attn_out, &[1, DIM]);

    // RMSNorm before FFN
    let rn2_w = b.add_input("rn2_w", &[DIM]);
    let rn2_eps = b.add_input("rn2_eps", &[1]);
    let normed2 = b.add_rms_norm(res1, rn2_eps, 1, rn2_w, &[1, DIM]);

    // SwiGLU FFN: gate + up -> SiLU(gate) * up -> down
    let gate_w = b.add_input("gate_w", &[FFN_DIM, DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, DIM]);
    let down_w = b.add_input("down_w", &[DIM, FFN_DIM]);
    let gate = b.add_linear(normed2, gate_w, None, &[1, FFN_DIM]);
    let up = b.add_linear(normed2, up_w, None, &[1, FFN_DIM]);
    let gate_act = add_silu(&mut b, gate, &[1, FFN_DIM]);
    let gated = b.add_binary_mul(gate_act, up, &[1, FFN_DIM]);
    let ffn_out = b.add_linear(gated, down_w, None, &[1, DIM]);

    // Final residual
    let result = b.add_binary_add(res1, ffn_out, &[1, DIM]);
    let def = b.build(result).expect("valid full auto step kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::Variable, // cached_k
        TensorParamBinding::Variable, // cached_v
        ones(&[DIM]),                 // rn1_w
        eps_binding(),                // rn1_eps
        weight(&[KV_DIM, DIM]),       // q_weight
        weight(&[DIM, KV_DIM]),       // out_weight
        ones(&[DIM]),                 // rn2_w
        eps_binding(),                // rn2_eps
        weight(&[FFN_DIM, DIM]),      // gate_w
        weight(&[FFN_DIM, DIM]),      // up_w
        weight(&[DIM, FFN_DIM]),      // down_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[3, cache_len, KV_DIM], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Full auto step IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Full auto step CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 18. Multi-layer KV cache depth composition (IBP + CROWN)
// ===========================================================================

#[test]
fn test_kv_cache_multi_layer_depth_ibp_crown() {
    // 2-layer decoder with separate KV caches per layer.
    // Layer 0: RMSNorm -> attention(Q, cached_K0, cached_V0) -> residual
    // Layer 1: RMSNorm -> attention(Q, cached_K1, cached_V1) -> residual
    let cache_len: usize = 4;

    let mut b = TensorBlockBuilder::new("kv_cache_multi_layer_depth");
    let input = b.add_input("x", &[1, DIM]);
    let cached_k0 = b.add_input("cached_k0", &[cache_len, KV_DIM]);
    let cached_v0 = b.add_input("cached_v0", &[cache_len, KV_DIM]);
    let cached_k1 = b.add_input("cached_k1", &[cache_len, KV_DIM]);
    let cached_v1 = b.add_input("cached_v1", &[cache_len, KV_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Layer 0
    let rn0_w = b.add_input("rn0_w", &[DIM]);
    let rn0_eps = b.add_input("rn0_eps", &[1]);
    let normed0 = b.add_rms_norm(input, rn0_eps, 1, rn0_w, &[1, DIM]);
    let q0_w = b.add_input("q0_weight", &[KV_DIM, DIM]);
    let q0 = b.add_linear(normed0, q0_w, None, &[1, KV_DIM]);
    let attn0 = b.add_attention(
        q0,
        cached_k0,
        cached_v0,
        AttentionMask::Standard,
        Some(scale),
        &[1, KV_DIM],
    );
    let out0_w = b.add_input("out0_weight", &[DIM, KV_DIM]);
    let attn0_out = b.add_linear(attn0, out0_w, None, &[1, DIM]);
    let res0 = b.add_binary_add(input, attn0_out, &[1, DIM]);

    // Layer 1
    let rn1_w = b.add_input("rn1_w", &[DIM]);
    let rn1_eps = b.add_input("rn1_eps", &[1]);
    let normed1 = b.add_rms_norm(res0, rn1_eps, 1, rn1_w, &[1, DIM]);
    let q1_w = b.add_input("q1_weight", &[KV_DIM, DIM]);
    let q1 = b.add_linear(normed1, q1_w, None, &[1, KV_DIM]);
    let attn1 = b.add_attention(
        q1,
        cached_k1,
        cached_v1,
        AttentionMask::Standard,
        Some(scale),
        &[1, KV_DIM],
    );
    let out1_w = b.add_input("out1_weight", &[DIM, KV_DIM]);
    let attn1_out = b.add_linear(attn1, out1_w, None, &[1, DIM]);
    let result = b.add_binary_add(res0, attn1_out, &[1, DIM]);

    let def = b.build(result).expect("valid multi-layer depth kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::Variable, // cached_k0
        TensorParamBinding::Variable, // cached_v0
        TensorParamBinding::Variable, // cached_k1
        TensorParamBinding::Variable, // cached_v1
        ones(&[DIM]),                 // rn0_w
        eps_binding(),                // rn0_eps
        weight(&[KV_DIM, DIM]),       // q0_weight
        weight(&[DIM, KV_DIM]),       // out0_weight
        ones(&[DIM]),                 // rn1_w
        eps_binding(),                // rn1_eps
        weight(&[KV_DIM, DIM]),       // q1_weight
        weight(&[DIM, KV_DIM]),       // out1_weight
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[5, cache_len, KV_DIM], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Multi-layer depth (2 layers) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Multi-layer depth (2 layers) CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}
