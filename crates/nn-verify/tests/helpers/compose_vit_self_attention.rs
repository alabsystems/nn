// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: ViT self-attention sub-block NY composition.
//!
//! Verifies bounds propagation through the self-attention sub-block in isolation:
//!   Q/K/V projections -> multi-head attention core -> output projection
//!
//! Architecture (Dosovitskiy et al. 2020 "An Image is Worth 16x16 Words"):
//! - Q, K, V are separate Linear projections from hidden_states
//! - Attention core: Q*K^T / sqrt(d_k) -> softmax -> * V
//! - Output projection: Linear from concatenated heads back to embed_dim
//! - ViT uses standard (bidirectional) attention, not causal
//!
//! Sub-blocks tested individually:
//! 1. Q/K/V projection: 3 separate Linear layers
//! 2. Attention core: scaled dot-product attention via `add_attention`
//! 3. Output projection: Linear from head-concatenated to embed_dim
//! 4. Full self-attention: all sub-blocks composed via `add_multi_head_attention`
//!
//! Dimensions: SEQ_LEN=4, EMBED_DIM=64, NUM_HEADS=4 (head_dim=16).
//!
//! Part of #3544: ViT self-attention NY compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions — small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Sequence length (number of patch tokens).
const SEQ_LEN: usize = 4;
/// Embedding dimension (tiny ViT hidden size).
const EMBED_DIM: usize = 64;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Per-head dimension: EMBED_DIM / NUM_HEADS.
const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS; // 16

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build a Q/K/V projection kernel: 3 separate Linear layers in parallel.
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable).
/// Output: `[SEQ_LEN, EMBED_DIM]` (V projection — all three have the same shape,
/// we output V as representative; Q and K are intermediate nodes in the graph).
///
/// This tests that bounds propagate correctly through multiple Linear projections
/// from a single variable input, which is the entry point of self-attention.
fn build_qkv_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vit_qkv_projection");

    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);

    let proj_shape = [SEQ_LEN, EMBED_DIM];

    // Three independent linear projections from the same input.
    // We concatenate Q, K, V via a stack-like pattern. Since TensorBlockBuilder
    // outputs a single node, we chain them: the final output captures V,
    // but Q and K are still computed in the graph (exercising the projections).
    let _q = b.add_linear(input, q_w, None, &proj_shape);
    let _k = b.add_linear(input, k_w, None, &proj_shape);
    let v = b.add_linear(input, v_w, None, &proj_shape);

    // Output the V projection. All three projections are structurally identical
    // Linear layers, so verifying V is representative. The graph still contains
    // all three projection nodes for structural completeness.
    b.build(v).expect("valid Q/K/V projection kernel")
}

/// Build the attention core kernel: scaled dot-product attention.
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable).
/// Internally: Q/K/V projections -> reshape -> transpose -> attention -> transpose -> reshape.
/// Output: `[SEQ_LEN, EMBED_DIM]` (attention output before output projection).
///
/// This uses `add_multi_head_attention` but WITHOUT the output projection weight,
/// so we manually decompose the steps to isolate the attention core.
fn build_attention_core_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vit_attention_core");

    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);

    let proj_shape = [SEQ_LEN, EMBED_DIM];

    // Project Q, K, V: [T, D] -> [T, D]
    let q = b.add_linear(input, q_w, None, &proj_shape);
    let k = b.add_linear(input, k_w, None, &proj_shape);
    let v = b.add_linear(input, v_w, None, &proj_shape);

    // Reshape to [T, H, head_dim]
    let reshaped = [SEQ_LEN, NUM_HEADS, HEAD_DIM];
    let q = b.add_reshape(q, &reshaped);
    let k = b.add_reshape(k, &reshaped);
    let v = b.add_reshape(v, &reshaped);

    // Transpose to [H, T, head_dim] for per-head attention
    let transposed = [NUM_HEADS, SEQ_LEN, HEAD_DIM];
    let q = b.add_transpose(q, &[1, 0, 2], &transposed);
    let k = b.add_transpose(k, &[1, 0, 2], &transposed);
    let v = b.add_transpose(v, &[1, 0, 2], &transposed);

    // Scaled dot-product attention: softmax(Q*K^T / sqrt(d_k)) * V
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &transposed);

    // Transpose back to [T, H, head_dim]
    let attn = b.add_transpose(attn, &[1, 0, 2], &reshaped);

    // Reshape back to [T, D]
    let out = b.add_reshape(attn, &proj_shape);

    b.build(out).expect("valid attention core kernel")
}

/// Build the output projection kernel: Linear from embed_dim to embed_dim.
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable — represents concatenated head output).
/// Output: `[SEQ_LEN, EMBED_DIM]`.
///
/// This isolates the final Linear layer that maps from concatenated attention
/// heads back to the model dimension. Trivially a single Linear — tests that
/// the projection preserves bounds correctly.
fn build_output_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vit_output_projection");

    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_b = b.add_input("out_bias", &[EMBED_DIM]);

    let out = b.add_linear(input, out_w, Some(out_b), &[SEQ_LEN, EMBED_DIM]);

    b.build(out).expect("valid output projection kernel")
}

/// Build the full self-attention sub-block using `add_multi_head_attention`.
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable).
/// Output: `[SEQ_LEN, EMBED_DIM]`.
///
/// Composes: Q/K/V projection -> attention core -> output projection.
/// Uses `add_multi_head_attention` which encapsulates the full pattern.
fn build_full_self_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vit_full_self_attention");

    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard, // ViT uses bidirectional attention
            &[SEQ_LEN, EMBED_DIM],
        )
        .expect("valid multi-head attention");

    b.build(out).expect("valid full self-attention kernel")
}

// ---------------------------------------------------------------------------
// Binding helpers
// ---------------------------------------------------------------------------

/// Bindings for Q/K/V projection kernel.
fn qkv_projection_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);

    vec![
        TensorParamBinding::Variable, // x [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj), // v_weight [D, D]
    ]
}

/// Bindings for attention core kernel.
fn attention_core_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);

    vec![
        TensorParamBinding::Variable, // x [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj), // v_weight [D, D]
    ]
}

/// Bindings for output projection kernel.
fn output_projection_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);
    let bias = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,               // x [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantTensor(w_proj), // out_weight [D, D]
        TensorParamBinding::ConstantTensor(bias),   // out_bias [D]
    ]
}

/// Bindings for full self-attention kernel.
fn full_self_attention_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);

    vec![
        TensorParamBinding::Variable, // x [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj), // out_weight [D, D]
    ]
}

// ---------------------------------------------------------------------------
// Q/K/V Projection tests
// ---------------------------------------------------------------------------

/// Q/K/V projection TensorKernelDef validates.
#[test]
fn test_vit_qkv_projection_def_validates() {
    let def = build_qkv_projection_kernel();
    def.validate()
        .expect("Q/K/V projection kernel should validate");
}

/// Q/K/V projection translates to NY GraphNetwork.
#[test]
fn test_vit_qkv_projection_graph_builds() {
    let def = build_qkv_projection_kernel();
    let bindings = qkv_projection_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("Q/K/V projection graph should translate");

    // 3 Linear projections = at least 3 nodes.
    assert!(
        graph.num_nodes() >= 3,
        "Q/K/V projection graph should have >= 3 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through Q/K/V projection.
#[test]
fn test_vit_qkv_projection_ibp_propagates() {
    let def = build_qkv_projection_kernel();
    let bindings = qkv_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Q/K/V projection");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT Q/K/V projection IBP: bounds=[{lo_min}, {hi_max}]");

    // Linear projection with 0.02 weights and [-1, 1] input.
    // Output magnitude: ~0.02 * 64 * 1.0 = ~1.28 per element.
    assert!(
        lo_min > -100.0,
        "IBP lower should be > -100 with small weights, got {lo_min}"
    );
    assert!(
        hi_max < 100.0,
        "IBP upper should be < 100 with small weights, got {hi_max}"
    );
}

/// CROWN bounds propagate through Q/K/V projection.
///
/// Linear layers propagate exactly through both IBP and CROWN.
/// CROWN should match or be tighter than IBP.
#[test]
fn test_vit_qkv_projection_crown_propagation() {
    let def = build_qkv_projection_kernel();
    let bindings = qkv_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT Q/K/V projection: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ---------------------------------------------------------------------------
// Attention Core tests
// ---------------------------------------------------------------------------

/// Attention core TensorKernelDef validates.
#[test]
fn test_vit_attention_core_def_validates() {
    let def = build_attention_core_kernel();
    def.validate()
        .expect("attention core kernel should validate");
}

/// Attention core translates to NY GraphNetwork.
#[test]
fn test_vit_attention_core_graph_builds() {
    let def = build_attention_core_kernel();
    let bindings = attention_core_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("attention core graph should translate");

    // 3 Linear + 3 Reshape + 3 Transpose + Attention + Transpose + Reshape = 13+ nodes.
    assert!(
        graph.num_nodes() >= 10,
        "attention core graph should have >= 10 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the attention core.
#[test]
fn test_vit_attention_core_ibp_propagates() {
    let def = build_attention_core_kernel();
    let bindings = attention_core_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through attention core");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT attention core IBP: bounds=[{lo_min}, {hi_max}]");

    // Attention output includes softmax normalization, so bounds are bounded.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN bounds propagate through the attention core.
///
/// Attention includes softmax (piecewise-smooth) and bilinear matmul (Q*K^T,
/// attn_weights*V). CROWN uses McCormick relaxation for bilinear terms and
/// piecewise linearization for softmax. May fall back to IBP on complex
/// attention structures.
#[test]
fn test_vit_attention_core_crown_propagation() {
    let def = build_attention_core_kernel();
    let bindings = attention_core_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT attention core: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

// ---------------------------------------------------------------------------
// Output Projection tests
// ---------------------------------------------------------------------------

/// Output projection TensorKernelDef validates.
#[test]
fn test_vit_output_projection_def_validates() {
    let def = build_output_projection_kernel();
    def.validate()
        .expect("output projection kernel should validate");
}

/// Output projection translates to NY GraphNetwork.
#[test]
fn test_vit_output_projection_graph_builds() {
    let def = build_output_projection_kernel();
    let bindings = output_projection_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("output projection graph should translate");

    // Single Linear with bias = at least 1 node.
    assert!(
        graph.num_nodes() >= 1,
        "output projection graph should have >= 1 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the output projection.
#[test]
fn test_vit_output_projection_ibp_propagates() {
    let def = build_output_projection_kernel();
    let bindings = output_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through output projection");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT output projection IBP: bounds=[{lo_min}, {hi_max}]");

    // Single linear layer with 0.02 weights, [-1, 1] input, zero bias.
    assert!(lo_min > -100.0, "IBP lower should be > -100, got {lo_min}");
    assert!(hi_max < 100.0, "IBP upper should be < 100, got {hi_max}");
}

/// CROWN bounds propagate through the output projection.
///
/// A single Linear layer propagates exactly in CROWN. This should always
/// succeed without fallback and produce identical bounds to IBP (Linear
/// is exact in both propagation methods).
#[test]
fn test_vit_output_projection_crown_propagation() {
    let def = build_output_projection_kernel();
    let bindings = output_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT output projection: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ---------------------------------------------------------------------------
// Full Self-Attention tests
// ---------------------------------------------------------------------------

/// Full self-attention TensorKernelDef validates.
#[test]
fn test_vit_full_self_attention_def_validates() {
    let def = build_full_self_attention_kernel();
    def.validate()
        .expect("full self-attention kernel should validate");
}

/// Full self-attention translates to NY GraphNetwork.
#[test]
fn test_vit_full_self_attention_graph_builds() {
    let def = build_full_self_attention_kernel();
    let bindings = full_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("full self-attention graph should translate");

    // 4 Linear (Q,K,V,out) + 3 Reshape + 3 Transpose + Attention + Transpose + Reshape
    // = 14+ nodes.
    assert!(
        graph.num_nodes() >= 10,
        "full self-attention graph should have >= 10 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through full self-attention.
#[test]
fn test_vit_full_self_attention_ibp_propagates() {
    let def = build_full_self_attention_kernel();
    let bindings = full_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full self-attention");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT full self-attention IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN bounds propagate through full self-attention.
///
/// Self-attention includes softmax and bilinear operations which require
/// CROWN linearization. May fall back to IBP due to attention complexity.
/// When CROWN succeeds, it should produce tighter bounds.
#[test]
fn test_vit_full_self_attention_crown_propagation() {
    let def = build_full_self_attention_kernel();
    let bindings = full_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT full self-attention: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Full self-attention verify and record under "vit_self_attention" key.
///
/// Self-attention does NOT contain LayerNorm (that is in the encoder block
/// wrapper), so soundness mode should be Sound (not Heuristic) unless
/// attention linearization triggers heuristic fallback.
#[test]
fn test_vit_self_attention_verify_and_record() {
    let def = build_full_self_attention_kernel();
    let bindings = full_self_attention_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "vit_self_attention");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, EMBED_DIM]);

    eprintln!(
        "vit_self_attention soundness_mode: {:?}",
        result.verification.soundness_mode
    );
}

// ---------------------------------------------------------------------------
// Bounds sanity tests
// ---------------------------------------------------------------------------

/// Q/K/V projection bounds width is reasonable for small weights.
#[test]
fn test_vit_qkv_projection_bounds_width() {
    let def = build_qkv_projection_kernel();
    let bindings = qkv_projection_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "vit_qkv_projection");

    let (lo_arr, hi_arr) = result.output_bounds.lower_upper();
    let diff = hi_arr.to_owned() - lo_arr.to_owned();
    let max_width = diff.iter().copied().fold(0.0f32, f32::max);
    eprintln!("vit_qkv_projection max bounds width: {max_width:.4}");
    // Linear with 0.02 weights on [-1, 1] input: width ~= 2 * 0.02 * 64 = 2.56.
    assert!(
        max_width < 50.0,
        "Q/K/V projection bounds width {max_width} should be < 50 for unit input range"
    );
}

/// Full self-attention all bounds are finite (no NaN/Inf from attention ops).
#[test]
fn test_vit_full_self_attention_bounds_finite() {
    let def = build_full_self_attention_kernel();
    let bindings = full_self_attention_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "vit_self_attention_finite");

    let (lo_arr, hi_arr) = result.output_bounds.lower_upper();
    for &v in lo_arr.iter() {
        assert!(v.is_finite(), "lower bound is not finite: {v}");
    }
    for &v in hi_arr.iter() {
        assert!(v.is_finite(), "upper bound is not finite: {v}");
    }
    for (lo, hi) in lo_arr.iter().zip(hi_arr.iter()) {
        assert!(lo <= hi, "lower {lo} > upper {hi}");
    }
}

/// Attention core with narrow input bounds produces tighter output bounds.
///
/// With [-0.1, 0.1] input (narrower than [-1, 1]), IBP output bounds should
/// be proportionally tighter for linear sub-blocks, and bounded for attention.
#[test]
fn test_vit_attention_core_narrow_input_tighter_bounds() {
    let def = build_attention_core_kernel();
    let bindings = attention_core_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);
    let narrow_input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 0.1);

    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");

    let (wide_lo, wide_hi) = bounds_min_max(&wide_output);
    let (narrow_lo, narrow_hi) = bounds_min_max(&narrow_output);

    let wide_width = wide_hi - wide_lo;
    let narrow_width = narrow_hi - narrow_lo;

    eprintln!("Wide input IBP width: {wide_width:.4}");
    eprintln!("Narrow input IBP width: {narrow_width:.4}");

    // Narrower input should produce equal or narrower output bounds.
    // (IBP is monotone in input width for most operations.)
    assert!(
        narrow_width <= wide_width + 1e-4,
        "narrow input ({narrow_width:.4}) should produce <= wide input ({wide_width:.4}) bounds"
    );
}
