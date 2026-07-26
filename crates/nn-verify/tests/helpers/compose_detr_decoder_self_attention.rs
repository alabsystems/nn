// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: DETR decoder self-attention sub-block NY composition.
//!
//! Verifies bounds propagation through the decoder self-attention sub-block
//! in isolation, before cross-attention with encoder features.
//!
//! Architecture (Carion et al. 2020):
//!   In the DETR decoder, the first sub-block is self-attention where learned
//!   object queries attend to each other. This is bidirectional (no causal mask)
//!   because object queries represent unordered detection slots.
//!
//!   Sub-block: LayerNorm(x) -> MHA(bidirectional, Q=K=V) -> + x (residual)
//!
//! This tests the sub-block in isolation at two sizes, validating that IBP
//! and CROWN produce finite, valid bounds through the attention mechanism.
//!
//! Part of #3556: DETR object detection compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding, VerificationSoundnessMode};
use ndarray::{ArrayD, IxDyn};

// ===========================================================================
// Small configuration: 4 object queries, d=32, 2 heads
// ===========================================================================

mod small {
    pub(super) const NUM_QUERIES: usize = 4;
    pub(super) const EMBED_DIM: usize = 32;
    pub(super) const NUM_HEADS: usize = 2;
}

// ===========================================================================
// Medium configuration: 10 object queries, d=64, 4 heads
// ===========================================================================

mod medium {
    pub(super) const NUM_QUERIES: usize = 10;
    pub(super) const EMBED_DIM: usize = 64;
    pub(super) const NUM_HEADS: usize = 4;
}

/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// Builder helpers
// ===========================================================================

/// Build a decoder self-attention sub-block: LN -> MHA(bidirectional) -> residual.
///
/// Input: `[num_queries, embed_dim]` (Variable -- learned object queries).
/// Output: `[num_queries, embed_dim]`.
///
/// Object queries attend to each other with bidirectional attention.
/// No causal mask: detection slots are unordered.
fn build_decoder_self_attention_kernel(
    name: &str,
    num_queries: usize,
    embed_dim: usize,
    num_heads: usize,
) -> TensorKernelDef {
    let d = embed_dim;
    let mut b = TensorBlockBuilder::new(name);

    let input = b.add_input("object_queries", &[num_queries, d]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[d]);
    let ln_b = b.add_input("ln_bias", &[d]);
    let q_w = b.add_input("q_weight", &[d, d]);
    let k_w = b.add_input("k_weight", &[d, d]);
    let v_w = b.add_input("v_weight", &[d, d]);
    let out_w = b.add_input("out_weight", &[d, d]);

    let shape = [num_queries, d];

    // Pre-norm: LayerNorm
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);

    // Multi-head self-attention (bidirectional)
    let attn = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            out_w,
            num_heads,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid self-attention");

    // Residual connection
    let out = b.add_binary_add(input, attn, &shape);

    b.build(out).expect("valid decoder self-attention kernel")
}

/// Bindings for the decoder self-attention sub-block.
fn decoder_self_attention_bindings(embed_dim: usize) -> Vec<TensorParamBinding> {
    let d = embed_dim;
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // object_queries [Q, D]
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ln_w), // ln_weight [D]
        TensorParamBinding::ConstantTensor(ln_b), // ln_bias [D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj), // out_weight [D, D]
    ]
}

// ===========================================================================
// Tests: Small configuration (4 queries, d=32, 2 heads)
// ===========================================================================

/// Decoder self-attention sub-block TensorKernelDef validates (small).
#[test]
fn test_detr_dec_self_attn_sub_small_def_validates() {
    let def = build_decoder_self_attention_kernel(
        "detr_dec_sa_sub_small",
        small::NUM_QUERIES,
        small::EMBED_DIM,
        small::NUM_HEADS,
    );
    def.validate()
        .expect("decoder self-attention sub-block (small) should validate");
}

/// Decoder self-attention graph builds (small).
#[test]
fn test_detr_dec_self_attn_sub_small_graph_builds() {
    let def = build_decoder_self_attention_kernel(
        "detr_dec_sa_sub_small_graph",
        small::NUM_QUERIES,
        small::EMBED_DIM,
        small::NUM_HEADS,
    );
    let bindings = decoder_self_attention_bindings(small::EMBED_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("decoder self-attention graph should translate");

    // LN + Q/K/V projections + attention + output projection + residual
    assert!(
        graph.num_nodes() >= 5,
        "decoder self-attention graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through decoder self-attention (small).
///
/// With bidirectional attention, all 4 object queries attend to each other.
/// Small weights (0.02) and [-1, 1] input should yield finite bounds.
#[test]
fn test_detr_dec_self_attn_sub_small_ibp_propagates() {
    let def = build_decoder_self_attention_kernel(
        "detr_dec_sa_sub_small_ibp",
        small::NUM_QUERIES,
        small::EMBED_DIM,
        small::NUM_HEADS,
    );
    let bindings = decoder_self_attention_bindings(small::EMBED_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[small::NUM_QUERIES, small::EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder self-attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[small::NUM_QUERIES, small::EMBED_DIM],
        "output shape must be [NUM_QUERIES, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder self-attn sub-block (small) IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through decoder self-attention (small).
///
/// LayerNorm requires heuristic CROWN linearization (IbpValidated mode).
#[test]
fn test_detr_dec_self_attn_sub_small_crown_propagation() {
    let def = build_decoder_self_attention_kernel(
        "detr_dec_sa_sub_small_crown",
        small::NUM_QUERIES,
        small::EMBED_DIM,
        small::NUM_HEADS,
    );
    let bindings = decoder_self_attention_bindings(small::EMBED_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[small::NUM_QUERIES, small::EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[small::NUM_QUERIES, small::EMBED_DIM],
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "DETR decoder self-attn sub-block (small): method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record decoder self-attention sub-block (small).
#[test]
fn test_detr_dec_self_attn_sub_small_verify_and_record() {
    let def = build_decoder_self_attention_kernel(
        "detr_decoder_self_attn_small",
        small::NUM_QUERIES,
        small::EMBED_DIM,
        small::NUM_HEADS,
    );
    let bindings = decoder_self_attention_bindings(small::EMBED_DIM);
    let input = uniform_bounds(&[small::NUM_QUERIES, small::EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_decoder_self_attn_small");
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (object queries)"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[small::NUM_QUERIES, small::EMBED_DIM]);

    // LayerNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "decoder self-attention with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: Medium configuration (10 queries, d=64, 4 heads)
// ===========================================================================

/// Decoder self-attention sub-block validates (medium).
#[test]
fn test_detr_dec_self_attn_sub_medium_def_validates() {
    let def = build_decoder_self_attention_kernel(
        "detr_dec_sa_sub_medium",
        medium::NUM_QUERIES,
        medium::EMBED_DIM,
        medium::NUM_HEADS,
    );
    def.validate()
        .expect("decoder self-attention (medium) should validate");
}

/// IBP bounds propagate through decoder self-attention (medium).
#[test]
fn test_detr_dec_self_attn_sub_medium_ibp_propagates() {
    let def = build_decoder_self_attention_kernel(
        "detr_dec_sa_sub_medium_ibp",
        medium::NUM_QUERIES,
        medium::EMBED_DIM,
        medium::NUM_HEADS,
    );
    let bindings = decoder_self_attention_bindings(medium::EMBED_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[medium::NUM_QUERIES, medium::EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder self-attention (medium)");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[medium::NUM_QUERIES, medium::EMBED_DIM],
        "output shape must be [NUM_QUERIES, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder self-attn sub-block (medium) IBP: bounds=[{lo_min}, {hi_max}]");
}
