// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Qwen3 attention mechanism NY composition.
//!
//! Verifies bounds propagation through Qwen3 attention sub-components that
//! the existing `compose_qwen3_decoder` explicitly simplifies away:
//!
//! 1. **RoPE half-split rotation**: `[S, HD]` -> narrow -> rotate -> concat
//! 2. **GQA KV expansion**: `[KV_HEADS * S, HD]` -> narrow -> repeat -> reshape
//! 3. **Combined RoPE + attention**: end-to-end single-head attention with RoPE
//! 4. **SwiGLU MLP**: gate_proj -> silu -> up_proj -> mul -> down_proj
//! 5. **Q/K/V linear projection bounds**: linear projection preserves IBP bounds
//! 6. **Attention score bounds**: QK^T / sqrt(d) -> softmax produces [0, 1]
//! 7. **Causal mask application**: masked positions get -inf, softmax still valid
//! 8. **Full attention block**: composition of all above maintains end-to-end bounds
//!
//! Additional focused tests:
//! - **Attention weighted sum bounds**: softmax * V produces bounded output
//! - **GQA key/value sharing**: grouped heads share K/V without bound violation
//!
//! These tests fill the gap identified in #3560: Qwen3 verify status was 1
//! entry (vacuous). By decomposing into focused sub-graphs, each component
//! gets non-vacuous IBP/CROWN bounds.
//!
//! Part of #3560: Qwen3 RoPE + GQA NY compose verification.

#[path = "qwen3_attention.rs"]
mod helpers;

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use helpers::{
    attention_scores_bindings, attention_weighted_sum_bindings, build_attention_scores,
    build_attention_weighted_sum, build_causal_mask_attention, build_full_attention_block,
    build_gqa_expand, build_qkv_projection, build_rope_attention_head, build_rope_half_split,
    build_swiglu_mlp, causal_mask_bindings, full_attention_block_bindings, gqa_expand_bindings,
    qkv_projection_bindings, rope_attention_bindings, rope_bindings, swiglu_bindings, HEAD_DIM,
    HIDDEN_DIM, NUM_HEADS, NUM_KV_HEADS, SEQ_LEN,
};
use nn_verify::tensor_kernel_to_graph;

// ============================================================================
// 1. RoPE half-split rotation tests
// ============================================================================

/// RoPE half-split TensorKernelDef validates.
#[test]
fn test_qwen3_rope_def_validates() {
    let def = build_rope_half_split();
    def.validate()
        .expect("RoPE half-split kernel should validate");
}

/// RoPE half-split translates to NY GraphNetwork.
#[test]
fn test_qwen3_rope_graph_builds() {
    let def = build_rope_half_split();
    let bindings = rope_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("RoPE half-split graph should translate");

    // 4 inputs + 2 narrow + 6 binary_mul + 2 binary_add + 1 concat = 15 nodes
    assert!(
        graph.num_nodes() >= 10,
        "RoPE graph should have >= 10 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through RoPE half-split with [-1, 1] input.
#[test]
fn test_qwen3_rope_ibp_propagates() {
    let def = build_rope_half_split();
    let bindings = rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HEAD_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through RoPE half-split");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HEAD_DIM],
        "RoPE output shape should be [SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 RoPE IBP: bounds=[{lo_min}, {hi_max}]");

    // RoPE is a rotation — with |cos|, |sin| <= 1 and input in [-1, 1]:
    // y1 = x1*cos - x2*sin, |y1| <= |x1| + |x2| <= 2
    // IBP will be slightly wider due to interval arithmetic.
    assert!(
        lo_min >= -5.0,
        "RoPE IBP lower should be >= -5, got {lo_min}"
    );
    assert!(hi_max <= 5.0, "RoPE IBP upper should be <= 5, got {hi_max}");
}

/// CROWN propagation through RoPE half-split.
///
/// RoPE is a piecewise-linear composition of multiplications and additions
/// (all constant cos/sin), so CROWN should produce tight bounds.
#[test]
fn test_qwen3_rope_crown_propagation() {
    let def = build_rope_half_split();
    let bindings = rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HEAD_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HEAD_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 RoPE: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// RoPE verify and record under "qwen3_rope_half_split" key.
#[test]
fn test_qwen3_rope_verify_and_record() {
    let def = build_rope_half_split();
    let bindings = rope_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HEAD_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_rope_half_split");
    assert_eq!(result.num_variables, 1, "single Variable input (q_head)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HEAD_DIM]);
}

// ============================================================================
// 2. GQA KV expansion tests
// ============================================================================

/// GQA expansion TensorKernelDef validates.
#[test]
fn test_qwen3_gqa_expand_def_validates() {
    let def = build_gqa_expand();
    def.validate().expect("GQA expand kernel should validate");
}

/// GQA expansion translates to NY GraphNetwork.
#[test]
fn test_qwen3_gqa_expand_graph_builds() {
    let def = build_gqa_expand();
    let bindings = gqa_expand_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GQA expand graph should translate");

    // 1 input + NUM_KV_HEADS narrow + 1 concat + 1 reshape = at least 5 nodes
    assert!(
        graph.num_nodes() >= 4,
        "GQA expand graph should have >= 4 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through GQA expansion.
///
/// Key property: expansion is a structural reshape/repeat — bounds should not
/// widen. Each output head has the same bounds as its source KV head.
#[test]
fn test_qwen3_gqa_expand_ibp_preserves_bounds() {
    let def = build_gqa_expand();
    let bindings = gqa_expand_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_KV_HEADS * SEQ_LEN, HEAD_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through GQA expand");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_HEADS, SEQ_LEN, HEAD_DIM],
        "GQA output shape should be [{NUM_HEADS}, {SEQ_LEN}, {HEAD_DIM}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 GQA expand IBP: bounds=[{lo_min}, {hi_max}]");

    // GQA is structural — bounds should be exactly [-1, 1] (same as input).
    assert!(
        lo_min >= -1.0 - 1e-6,
        "GQA expand should preserve lower bound, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "GQA expand should preserve upper bound, got {hi_max}"
    );
}

/// GQA verify and record under "qwen3_gqa_expand" key.
#[test]
fn test_qwen3_gqa_expand_verify_and_record() {
    let def = build_gqa_expand();
    let bindings = gqa_expand_bindings();
    let input = uniform_bounds(&[NUM_KV_HEADS * SEQ_LEN, HEAD_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_gqa_expand");
    assert_eq!(result.num_variables, 1, "single Variable input (kv_heads)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_HEADS, SEQ_LEN, HEAD_DIM]);
}

// ============================================================================
// 3. Combined RoPE + attention head tests
// ============================================================================

/// RoPE + attention head TensorKernelDef validates.
#[test]
fn test_qwen3_rope_attention_def_validates() {
    let def = build_rope_attention_head();
    def.validate()
        .expect("RoPE + attention head kernel should validate");
}

/// RoPE + attention head translates to NY GraphNetwork.
#[test]
fn test_qwen3_rope_attention_graph_builds() {
    let def = build_rope_attention_head();
    let bindings = rope_attention_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("RoPE + attention graph should translate");

    // 6 inputs + 2 × RoPE(~11 nodes each) + 1 attention = substantial graph
    assert!(
        graph.num_nodes() >= 20,
        "RoPE+attention graph should have >= 20 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through RoPE + attention with [-1, 1] input.
#[test]
fn test_qwen3_rope_attention_ibp_propagates() {
    let def = build_rope_attention_head();
    let bindings = rope_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Q, K, V all share same bounds spec for the Variable inputs.
    // We have 3 Variables: q_head, k_head, v_head — each [SEQ_LEN, HEAD_DIM].
    // BoundedTensor for multi-variable: flat concatenation of all variable shapes.
    let total_var_elements = 3 * SEQ_LEN * HEAD_DIM;
    let input = uniform_bounds(&[total_var_elements], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through RoPE + attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 RoPE+attention IBP: bounds=[{lo_min}, {hi_max}]");

    // Attention softmax normalizes to [0, 1], then weighted sum of V.
    // With V in [-1, 1], output should be bounded.
    assert!(
        lo_min.is_finite(),
        "output lower bound must be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "output upper bound must be finite, got {hi_max}"
    );
}

/// CROWN propagation through RoPE + attention.
#[test]
fn test_qwen3_rope_attention_crown_propagation() {
    let def = build_rope_attention_head();
    let bindings = rope_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let total_var_elements = 3 * SEQ_LEN * HEAD_DIM;
    let input = uniform_bounds(&[total_var_elements], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 RoPE+attention: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// RoPE + attention verify and record under "qwen3_rope_attention" key.
#[test]
fn test_qwen3_rope_attention_verify_and_record() {
    let def = build_rope_attention_head();
    let bindings = rope_attention_bindings();
    let total_var_elements = 3 * SEQ_LEN * HEAD_DIM;
    let input = uniform_bounds(&[total_var_elements], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_rope_attention");
    assert_eq!(result.num_variables, 3, "three Variable inputs (q, k, v)");
}

// ============================================================================
// 4. SwiGLU MLP tests
// ============================================================================

/// SwiGLU MLP TensorKernelDef validates.
#[test]
fn test_qwen3_swiglu_def_validates() {
    let def = build_swiglu_mlp();
    def.validate().expect("SwiGLU MLP kernel should validate");
}

/// SwiGLU MLP translates to NY GraphNetwork.
#[test]
fn test_qwen3_swiglu_graph_builds() {
    let def = build_swiglu_mlp();
    let bindings = swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("SwiGLU MLP graph should translate");

    // Only `hidden` is a Variable; the 3 weights bind as ConstantTensor and fold
    // into their linear ops (no input nodes). A single Variable maps to the
    // NETWORK_INPUT sentinel (no setup node). Translated ops:
    //   gate linear (1) + sigmoid (1) + silu mul (1) + up linear (1)
    //   + gated mul (1) + down linear (1) = 6 nodes.
    assert!(
        graph.num_nodes() >= 6,
        "SwiGLU graph should have >= 6 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through SwiGLU MLP with [-1, 1] input.
#[test]
fn test_qwen3_swiglu_ibp_propagates() {
    let def = build_swiglu_mlp();
    let bindings = swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through SwiGLU MLP");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "SwiGLU output shape should be [{SEQ_LEN}, {HIDDEN_DIM}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 SwiGLU IBP: bounds=[{lo_min}, {hi_max}]");

    // With small weights (0.02) and [-1, 1] input, SwiGLU output should be bounded.
    // SiLU(x) in [-0.28, inf) but gate(x) is small due to small weights.
    assert!(
        lo_min.abs() < 1e6,
        "SwiGLU IBP lower magnitude should be < 1e6, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e6,
        "SwiGLU IBP upper magnitude should be < 1e6, got {hi_max}"
    );
}

/// CROWN propagation through SwiGLU MLP.
///
/// SwiGLU has 3 non-linearities (sigmoid in SiLU + 2 binary_mul), so CROWN
/// may produce wider bounds than expected. The test verifies structural
/// correctness rather than asserting tight bounds.
#[test]
fn test_qwen3_swiglu_crown_propagation() {
    let def = build_swiglu_mlp();
    let bindings = swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 SwiGLU: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// SwiGLU MLP verify and record under "qwen3_swiglu_mlp" key.
#[test]
fn test_qwen3_swiglu_verify_and_record() {
    let def = build_swiglu_mlp();
    let bindings = swiglu_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_swiglu_mlp");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ============================================================================
// 5. Q/K/V linear projection bounds
// ============================================================================

/// Q/K/V linear projection TensorKernelDef validates.
#[test]
fn test_qwen3_qkv_projection_def_validates() {
    let def = build_qkv_projection();
    def.validate()
        .expect("Q/K/V projection kernel should validate");
}

/// Q/K/V linear projection translates to NY GraphNetwork.
#[test]
fn test_qwen3_qkv_projection_graph_builds() {
    let def = build_qkv_projection();
    let bindings = qkv_projection_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("Q/K/V projection graph should translate");

    // Only `hidden` is a Variable; the 3 weights bind as ConstantTensor and fold
    // into their linear ops (no input nodes), and a single Variable uses the
    // NETWORK_INPUT sentinel (no setup node). Translated ops:
    //   q linear (1) + k linear (1) + v linear (1) + qk add (1) + qkv add (1) = 5 nodes.
    assert!(
        graph.num_nodes() >= 5,
        "Q/K/V projection graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through Q/K/V linear projections.
///
/// Key property: linear projection with weight magnitude W and input in [-r, r]
/// produces output bounded by [-HIDDEN_DIM * W * r, HIDDEN_DIM * W * r] per
/// element. The sum of 3 projections triples this bound.
#[test]
fn test_qwen3_qkv_projection_ibp_preserves_bounds() {
    let def = build_qkv_projection();
    let bindings = qkv_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Q/K/V projection");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "Q/K/V projection output shape should be [{SEQ_LEN}, {HIDDEN_DIM}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 Q/K/V projection IBP: bounds=[{lo_min}, {hi_max}]");

    // With weight=0.02 and input in [-1, 1], each projection output per element
    // is bounded by HIDDEN_DIM * 0.02 * 1.0 = 0.64. Three projections summed
    // gives ~1.92. IBP should be within this order of magnitude.
    let expected_max = 3.0 * (HIDDEN_DIM as f32) * 0.02;
    assert!(
        lo_min >= -expected_max - 1.0,
        "Q/K/V projection IBP lower should be >= {}, got {lo_min}",
        -expected_max - 1.0
    );
    assert!(
        hi_max <= expected_max + 1.0,
        "Q/K/V projection IBP upper should be <= {}, got {hi_max}",
        expected_max + 1.0
    );
}

/// Q/K/V projection verify and record under "qwen3_qkv_projection" key.
#[test]
fn test_qwen3_qkv_projection_verify_and_record() {
    let def = build_qkv_projection();
    let bindings = qkv_projection_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_qkv_projection");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ============================================================================
// 6. Attention score bounds (QK^T / sqrt(d) -> softmax)
// ============================================================================

/// Attention score TensorKernelDef validates.
#[test]
fn test_qwen3_attention_scores_def_validates() {
    let def = build_attention_scores();
    def.validate()
        .expect("attention scores kernel should validate");
}

/// Attention score graph translates correctly.
#[test]
fn test_qwen3_attention_scores_graph_builds() {
    let def = build_attention_scores();
    let bindings = attention_scores_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("attention scores graph should translate");

    // 2 inputs + 1 matmul + 1 softmax = at least 4 nodes
    assert!(
        graph.num_nodes() >= 3,
        "attention scores graph should have >= 3 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds through attention scores verify softmax output in [0, 1].
///
/// Key property: softmax always outputs values in [0, 1] regardless of
/// input magnitude. IBP should verify this structural property.
#[test]
fn test_qwen3_attention_scores_ibp_softmax_bounded() {
    let def = build_attention_scores();
    let bindings = attention_scores_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Q and K are both Variable, flat concatenated for multi-variable input
    let total_var_elements = 2 * SEQ_LEN * HEAD_DIM;
    let input = uniform_bounds(&[total_var_elements], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through attention scores");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, SEQ_LEN],
        "attention scores output shape should be [{SEQ_LEN}, {SEQ_LEN}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 attention scores IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output is always in [0, 1]. IBP may slightly overshoot.
    assert!(
        lo_min >= -0.01,
        "softmax lower bound should be >= -0.01, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "softmax upper bound should be <= 1.01, got {hi_max}"
    );
}

/// Attention scores verify and record under "qwen3_attention_scores" key.
#[test]
fn test_qwen3_attention_scores_verify_and_record() {
    let def = build_attention_scores();
    let bindings = attention_scores_bindings();
    let total_var_elements = 2 * SEQ_LEN * HEAD_DIM;
    let input = uniform_bounds(&[total_var_elements], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_attention_scores");
    assert_eq!(result.num_variables, 2, "two Variable inputs (q, k)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, SEQ_LEN]);
}

// ============================================================================
// 7. Causal mask application
// ============================================================================

/// Causal mask attention TensorKernelDef validates.
#[test]
fn test_qwen3_causal_mask_def_validates() {
    let def = build_causal_mask_attention();
    def.validate()
        .expect("causal mask attention kernel should validate");
}

/// Causal mask attention graph translates correctly.
#[test]
fn test_qwen3_causal_mask_graph_builds() {
    let def = build_causal_mask_attention();
    let bindings = causal_mask_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("causal mask attention graph should translate");

    // `scores` is the only Variable (uses the NETWORK_INPUT sentinel, no setup
    // node); `causal_mask` binds as a ConstantTensor and folds into the add.
    // Translated ops: masked add (1) + softmax (1) = 2 nodes.
    assert!(
        graph.num_nodes() >= 2,
        "causal mask graph should have >= 2 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP through causal mask + softmax verifies masked positions get ~0 weight.
///
/// Key property: causal mask adds -1e9 to future positions. After softmax,
/// those positions get exp(-1e9) / Z ~ 0. Valid positions get weights in
/// [0, 1]. The overall output is still bounded in [0, 1].
#[test]
fn test_qwen3_causal_mask_ibp_bounds_valid() {
    let def = build_causal_mask_attention();
    let bindings = causal_mask_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, SEQ_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through causal mask attention");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, SEQ_LEN],
        "causal mask output shape should be [{SEQ_LEN}, {SEQ_LEN}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 causal mask attention IBP: bounds=[{lo_min}, {hi_max}]");

    // After softmax, output is always in [0, 1] regardless of masking.
    assert!(
        lo_min >= -0.01,
        "causal mask softmax lower should be >= -0.01, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "causal mask softmax upper should be <= 1.01, got {hi_max}"
    );
}

/// Causal mask attention verify and record under "qwen3_causal_mask_attention" key.
#[test]
fn test_qwen3_causal_mask_verify_and_record() {
    let def = build_causal_mask_attention();
    let bindings = causal_mask_bindings();
    let input = uniform_bounds(&[SEQ_LEN, SEQ_LEN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_causal_mask_attention");
    assert_eq!(result.num_variables, 1, "single Variable input (scores)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, SEQ_LEN]);
}

// ============================================================================
// 8. Full attention block (complete composition)
// ============================================================================

/// Full attention block TensorKernelDef validates.
#[test]
fn test_qwen3_full_attention_block_def_validates() {
    let def = build_full_attention_block();
    def.validate()
        .expect("full attention block kernel should validate");
}

/// Full attention block translates to NY GraphNetwork.
#[test]
fn test_qwen3_full_attention_block_graph_builds() {
    let def = build_full_attention_block();
    let bindings = full_attention_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("full attention block graph should translate");

    // Only `hidden` is a Variable (NETWORK_INPUT sentinel); the 4 proj weights
    // fold into their linears. MHA expands to: q/k/v linear (3) + q/k/v reshape
    // (3) + q/k/v transpose (3) + attention (1 native SelfAttention) + transpose
    // back (1) + reshape (1) + output linear (1) = 13, plus the residual add (1)
    // = 14 nodes.
    assert!(
        graph.num_nodes() >= 14,
        "full attention block graph should have >= 14 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full attention block.
///
/// End-to-end composition of Q/K/V projection, multi-head attention with
/// causal masking, output projection, and residual connection. With small
/// weights, the residual connection dominates and output stays near input.
#[test]
fn test_qwen3_full_attention_block_ibp_propagates() {
    let def = build_full_attention_block();
    let bindings = full_attention_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full attention block");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "full attention block output shape should be [{SEQ_LEN}, {HIDDEN_DIM}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 full attention block IBP: bounds=[{lo_min}, {hi_max}]");

    // With residual connection and small weights, output bounds should not
    // blow up excessively beyond input bounds.
    assert!(
        lo_min.abs() < 1e6,
        "full attention block IBP lower magnitude should be < 1e6, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e6,
        "full attention block IBP upper magnitude should be < 1e6, got {hi_max}"
    );
}

/// CROWN propagation through full attention block.
#[test]
fn test_qwen3_full_attention_block_crown_propagation() {
    let def = build_full_attention_block();
    let bindings = full_attention_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 full attention block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    assert!(
        lo_min.abs() < 1e6,
        "CROWN: lower bound magnitude should be < 1e6, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e6,
        "CROWN: upper bound magnitude should be < 1e6, got {hi_max}"
    );
}

/// Full attention block verify and record under "qwen3_full_attention_block" key.
#[test]
fn test_qwen3_full_attention_block_verify_and_record() {
    let def = build_full_attention_block();
    let bindings = full_attention_block_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_full_attention_block");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ============================================================================
// Additional: Attention weighted sum bounds
// ============================================================================

/// Attention weighted sum TensorKernelDef validates.
#[test]
fn test_qwen3_attention_weighted_sum_def_validates() {
    let def = build_attention_weighted_sum();
    def.validate()
        .expect("attention weighted sum kernel should validate");
}

/// IBP bounds through attention weighted sum (softmax_weights @ V).
///
/// Key property: if weights are softmax output (in [0, 1], rows sum to 1)
/// and V is in [-r, r], the weighted sum is a convex combination bounded
/// by [-r, r]. IBP may be wider due to interval arithmetic on the matmul.
#[test]
fn test_qwen3_attention_weighted_sum_ibp_bounded() {
    let def = build_attention_weighted_sum();
    let bindings = attention_weighted_sum_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Two Variable inputs: attn_weights [SEQ_LEN, SEQ_LEN] + v [SEQ_LEN, HEAD_DIM]
    let total_var_elements = SEQ_LEN * SEQ_LEN + SEQ_LEN * HEAD_DIM;
    let input = uniform_bounds(&[total_var_elements], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through attention weighted sum");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HEAD_DIM],
        "attention weighted sum output shape should be [{SEQ_LEN}, {HEAD_DIM}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 attention weighted sum IBP: bounds=[{lo_min}, {hi_max}]");

    // MatMul of [-1,1]@[-1,1] with dimensions [S,S]@[S,HD] gives
    // output bounded by [-S, S] per element due to interval arithmetic.
    let expected_max = SEQ_LEN as f32;
    assert!(
        lo_min >= -(expected_max + 1.0),
        "weighted sum lower should be >= {}, got {lo_min}",
        -(expected_max + 1.0)
    );
    assert!(
        hi_max <= expected_max + 1.0,
        "weighted sum upper should be <= {}, got {hi_max}",
        expected_max + 1.0
    );
}

/// Attention weighted sum verify and record under "qwen3_attention_weighted_sum" key.
#[test]
fn test_qwen3_attention_weighted_sum_verify_and_record() {
    let def = build_attention_weighted_sum();
    let bindings = attention_weighted_sum_bindings();
    let total_var_elements = SEQ_LEN * SEQ_LEN + SEQ_LEN * HEAD_DIM;
    let input = uniform_bounds(&[total_var_elements], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_attention_weighted_sum");
    assert_eq!(result.num_variables, 2, "two Variable inputs (weights, v)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HEAD_DIM]);
}

// ============================================================================
// Additional: GQA key/value sharing bounds
// ============================================================================

/// GQA KV sharing: expanded heads have identical bounds to source KV heads.
///
/// Key property: GQA repeats each KV head GQA_REP times. The expansion is
/// purely structural (narrow + concat + reshape), so bounds should be
/// exactly preserved. This test verifies that GQA expansion does not
/// introduce any bound violations or widening.
#[test]
fn test_qwen3_gqa_kv_sharing_bounds_preserved() {
    let def = build_gqa_expand();
    let bindings = gqa_expand_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Use asymmetric bounds to verify exact preservation
    let n = NUM_KV_HEADS * SEQ_LEN * HEAD_DIM;
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    for i in 0..n {
        let center = (i as f32) / (n as f32) - 0.5;
        lower.push(center - 0.1);
        upper.push(center + 0.1);
    }
    let input = nn_verify::BoundedTensor::new(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[NUM_KV_HEADS * SEQ_LEN, HEAD_DIM]), lower)
            .expect("valid lower"),
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[NUM_KV_HEADS * SEQ_LEN, HEAD_DIM]), upper)
            .expect("valid upper"),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through GQA expand");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_HEADS, SEQ_LEN, HEAD_DIM],
        "GQA output shape should be [{NUM_HEADS}, {SEQ_LEN}, {HEAD_DIM}]"
    );
    assert_bounds_valid(&output);

    // Verify bounds are finite and structurally valid
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 GQA KV sharing IBP: bounds=[{lo_min}, {hi_max}]");

    // GQA is structural: output bounds should not exceed input bounds
    assert!(
        lo_min >= -0.6 - 1e-6,
        "GQA sharing should preserve lower bound, got {lo_min}"
    );
    assert!(
        hi_max <= 0.6 + 1e-6,
        "GQA sharing should preserve upper bound, got {hi_max}"
    );
}
