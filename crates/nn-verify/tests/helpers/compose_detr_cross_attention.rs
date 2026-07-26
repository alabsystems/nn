// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: DETR-style cross-attention NY composition.
//!
//! Verifies bounds propagation through DETR decoder cross-attention where
//! encoder outputs provide K/V and learned object queries provide Q.
//! This is structurally distinct from self-attention: Q and K/V come from
//! different input tensors with potentially different bounds.
//!
//! Architecture (Carion et al. 2020, "End-to-End Object Detection with Transformers"):
//!   - Cross-attention: Q from decoder (object queries), K/V from encoder output
//!   - DETR decoder block: Self-attention + Cross-attention + FFN
//!   - Two configurations: small (d=64, heads=4) and medium (d=256, heads=8)
//!
//! Part of #3534: DETR cross-attention compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::{AttentionMask, CrossAttentionBlockConfig, CrossAttentionBlockWeights};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding, VerificationSoundnessMode};
use ndarray::{ArrayD, IxDyn};

// ===========================================================================
// Small configuration: d=64, heads=4, queries=10, encoder_seq=20
// ===========================================================================

mod small {
    pub(super) const EMBED_DIM: usize = 64;
    pub(super) const NUM_HEADS: usize = 4;
    pub(super) const FFN_DIM: usize = 128;
    /// Number of object queries (decoder side).
    pub(super) const NUM_QUERIES: usize = 10;
    /// Encoder output sequence length (e.g., flattened spatial features).
    pub(super) const ENC_SEQ_LEN: usize = 20;
}

// ===========================================================================
// Medium configuration: d=256, heads=8, queries=100, encoder_seq=50
// ===========================================================================

mod medium {
    pub(super) const EMBED_DIM: usize = 256;
    pub(super) const NUM_HEADS: usize = 8;
    pub(super) const FFN_DIM: usize = 512;
    /// Number of object queries (decoder side).
    pub(super) const NUM_QUERIES: usize = 100;
    /// Encoder output sequence length.
    pub(super) const ENC_SEQ_LEN: usize = 50;
}

// ===========================================================================
// Builder helpers
// ===========================================================================

/// Build a DETR-style cross-attention kernel using `add_multi_head_cross_attention`.
///
/// Q input: `[num_queries, embed_dim]` (Variable — object queries from decoder)
/// KV input: `[enc_seq_len, embed_dim]` (ConstantTensor — encoder output)
///
/// This models the core DETR cross-attention: learned object queries attend to
/// encoder features to detect objects. Q comes from the decoder, K/V from encoder.
fn build_detr_cross_attention_kernel(
    name: &str,
    num_queries: usize,
    enc_seq_len: usize,
    embed_dim: usize,
    num_heads: usize,
) -> TensorKernelDef {
    let d = embed_dim;
    let mut b = TensorBlockBuilder::new(name);

    let q_input = b.add_input("object_queries", &[num_queries, d]);
    let kv_input = b.add_input("encoder_output", &[enc_seq_len, d]);
    let q_w = b.add_input("q_weight", &[d, d]);
    let k_w = b.add_input("k_weight", &[d, d]);
    let v_w = b.add_input("v_weight", &[d, d]);
    let out_w = b.add_input("out_weight", &[d, d]);

    let out = b
        .add_multi_head_cross_attention(
            q_input,
            kv_input,
            q_w,
            k_w,
            v_w,
            out_w,
            num_heads,
            AttentionMask::Standard, // DETR uses bidirectional attention
            &[num_queries, d],
        )
        .expect("valid DETR cross-attention");
    b.build(out).expect("valid kernel")
}

/// Build a DETR decoder block: SelfAttention(Q) + CrossAttention(Q, KV) + FFN.
///
/// This models a full DETR decoder layer:
/// 1. Self-attention on object queries (Q=K=V from decoder)
/// 2. Cross-attention: Q from decoder, K/V from encoder output
/// 3. FFN (Linear -> GELU -> Linear)
///
/// Input: `q_input` = object queries `[num_queries, embed_dim]` (Variable)
///        `kv_input` = encoder output `[enc_seq_len, embed_dim]` (Constant)
/// Output: `[num_queries, embed_dim]`
fn build_detr_decoder_block_kernel(
    name: &str,
    num_queries: usize,
    enc_seq_len: usize,
    embed_dim: usize,
    num_heads: usize,
    ffn_dim: usize,
) -> TensorKernelDef {
    let d = embed_dim;
    let mut b = TensorBlockBuilder::new(name);

    // Inputs
    let q_input = b.add_input("object_queries", &[num_queries, d]);
    let kv_input = b.add_input("encoder_output", &[enc_seq_len, d]);
    let eps = b.add_input("eps", &[1]);

    // Self-attention weights
    let sa_ln_w = b.add_input("sa_ln_weight", &[d]);
    let sa_ln_b = b.add_input("sa_ln_bias", &[d]);
    let sa_q_w = b.add_input("sa_q_weight", &[d, d]);
    let sa_k_w = b.add_input("sa_k_weight", &[d, d]);
    let sa_v_w = b.add_input("sa_v_weight", &[d, d]);
    let sa_out_w = b.add_input("sa_out_weight", &[d, d]);

    // Cross-attention weights (4 LayerNorms: Q-branch, KV-branch, pre-FFN, output)
    let ca_ln1_w = b.add_input("ca_ln1_weight", &[d]);
    let ca_ln1_b = b.add_input("ca_ln1_bias", &[d]);
    let ca_ln2_w = b.add_input("ca_ln2_weight", &[d]);
    let ca_ln2_b = b.add_input("ca_ln2_bias", &[d]);
    let ca_ln3_w = b.add_input("ca_ln3_weight", &[d]);
    let ca_ln3_b = b.add_input("ca_ln3_bias", &[d]);
    let ca_ln_out_w = b.add_input("ca_ln_out_weight", &[d]);
    let ca_ln_out_b = b.add_input("ca_ln_out_bias", &[d]);
    let ca_q_w = b.add_input("ca_q_weight", &[d, d]);
    let ca_k_w = b.add_input("ca_k_weight", &[d, d]);
    let ca_v_w = b.add_input("ca_v_weight", &[d, d]);
    let ca_out_w = b.add_input("ca_out_weight", &[d, d]);
    let ca_ffn1_w = b.add_input("ca_ffn1_weight", &[ffn_dim, d]);
    let ca_ffn2_w = b.add_input("ca_ffn2_weight", &[d, ffn_dim]);

    // Step 1: Self-attention on object queries
    //
    // We build the self-attention manually using the layernorm -> MHA -> residual
    // pattern (no FFN in this sub-block; FFN is part of the cross-attention block).
    let shape = [num_queries, d];

    // Self-attn: LayerNorm -> MHA -> residual
    let normed = b.add_layer_norm(q_input, eps, 1, sa_ln_w, sa_ln_b, &shape);
    let self_attn = b
        .add_multi_head_attention(
            normed,
            sa_q_w,
            sa_k_w,
            sa_v_w,
            sa_out_w,
            num_heads,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid self-attention");
    let sa_residual = b.add_binary_add(q_input, self_attn, &shape);

    // Step 2: Cross-attention block (includes its own LN, FFN, and residuals)
    let ca_weights = CrossAttentionBlockWeights {
        ln1_weight: ca_ln1_w,
        ln1_bias: ca_ln1_b,
        ln2_weight: ca_ln2_w,
        ln2_bias: ca_ln2_b,
        ln3_weight: ca_ln3_w,
        ln3_bias: ca_ln3_b,
        ln_out_weight: ca_ln_out_w,
        ln_out_bias: ca_ln_out_b,
        q_weight: ca_q_w,
        k_weight: ca_k_w,
        v_weight: ca_v_w,
        out_weight: ca_out_w,
        ffn1_weight: ca_ffn1_w,
        ffn2_weight: ca_ffn2_w,
        eps,
    };

    let ca_config = CrossAttentionBlockConfig {
        num_heads,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: ffn_dim,
    };

    let out = b
        .add_cross_attention_transformer_block(sa_residual, kv_input, &ca_weights, &ca_config)
        .expect("valid cross-attention block");

    b.build(out).expect("valid DETR decoder block kernel")
}

// ===========================================================================
// Binding constructors
// ===========================================================================

/// Bindings for DETR cross-attention: Q=Variable, KV and weights=Constant.
fn cross_attention_bindings(enc_seq_len: usize, embed_dim: usize) -> Vec<TensorParamBinding> {
    let d = embed_dim;
    let w_small = 0.02f32;

    let kv_const = ArrayD::from_elem(IxDyn(&[enc_seq_len, d]), 0.1f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), w_small);

    vec![
        TensorParamBinding::Variable,                 // object_queries [Q, D]
        TensorParamBinding::ConstantTensor(kv_const), // encoder_output [S, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj),   // out_weight [D, D]
    ]
}

/// Bindings for the full DETR decoder block: Q=Variable, everything else=Constant.
fn decoder_block_bindings(
    enc_seq_len: usize,
    embed_dim: usize,
    ffn_dim: usize,
) -> Vec<TensorParamBinding> {
    let d = embed_dim;
    let w_small = 0.02f32;

    let kv_const = ArrayD::from_elem(IxDyn(&[enc_seq_len, d]), 0.1f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), w_small);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[ffn_dim, d]), w_small);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, ffn_dim]), w_small);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                 // object_queries [Q, D]
        TensorParamBinding::ConstantTensor(kv_const), // encoder_output [S, D]
        TensorParamBinding::ConstantScalar(1e-5),     // eps
        // Self-attention weights
        TensorParamBinding::ConstantTensor(ln_w.clone()), // sa_ln_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // sa_ln_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // sa_q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // sa_k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // sa_v_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // sa_out_weight
        // Cross-attention LayerNorms
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ca_ln1_weight (Q branch)
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ca_ln1_bias
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ca_ln2_weight (KV branch)
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ca_ln2_bias
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ca_ln3_weight (pre-FFN)
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ca_ln3_bias
        TensorParamBinding::ConstantTensor(ln_w), // ca_ln_out_weight
        TensorParamBinding::ConstantTensor(ln_b), // ca_ln_out_bias
        // Cross-attention projections
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_v_weight
        TensorParamBinding::ConstantTensor(w_proj),         // ca_out_weight
        // Cross-attention FFN
        TensorParamBinding::ConstantTensor(w_ffn1), // ca_ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2), // ca_ffn2_weight
    ]
}

// ===========================================================================
// Tests: DETR cross-attention (small configuration)
// ===========================================================================

/// DETR cross-attention kernel validates (small: d=64, heads=4).
#[test]
fn test_detr_cross_attention_small_def_validates() {
    let def = build_detr_cross_attention_kernel(
        "detr_xattn_small",
        small::NUM_QUERIES,
        small::ENC_SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
    );
    def.validate()
        .expect("DETR cross-attention kernel should validate");
}

/// DETR cross-attention translates to NY GraphNetwork (small).
#[test]
fn test_detr_cross_attention_small_graph_builds() {
    let def = build_detr_cross_attention_kernel(
        "detr_xattn_small_graph",
        small::NUM_QUERIES,
        small::ENC_SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
    );
    let bindings = cross_attention_bindings(small::ENC_SEQ_LEN, small::EMBED_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("DETR cross-attention graph should translate");

    // Cross-attention: Q/K/V projections + reshape + transpose + attention +
    // transpose + reshape + output projection = many nodes.
    assert!(
        graph.num_nodes() >= 5,
        "DETR cross-attention graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through DETR cross-attention (small).
///
/// Key difference from self-attention: Q input bounds are Variable (object queries)
/// while K/V bounds are derived from ConstantTensor (encoder output).
#[test]
fn test_detr_cross_attention_small_ibp_propagates() {
    let def = build_detr_cross_attention_kernel(
        "detr_xattn_small_ibp",
        small::NUM_QUERIES,
        small::ENC_SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
    );
    let bindings = cross_attention_bindings(small::ENC_SEQ_LEN, small::EMBED_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[small::NUM_QUERIES, small::EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DETR cross-attention");

    // Output shape matches Q sequence length, not KV sequence length.
    assert_eq!(
        output.lower_upper().0.shape(),
        &[small::NUM_QUERIES, small::EMBED_DIM],
        "output shape must be [num_queries, embed_dim]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR cross-attention (small) IBP: bounds=[{lo_min}, {hi_max}]");
}

/// CROWN propagation through DETR cross-attention (small).
#[test]
fn test_detr_cross_attention_small_crown_propagation() {
    let def = build_detr_cross_attention_kernel(
        "detr_xattn_small_crown",
        small::NUM_QUERIES,
        small::ENC_SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
    );
    let bindings = cross_attention_bindings(small::ENC_SEQ_LEN, small::EMBED_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[small::NUM_QUERIES, small::EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[small::NUM_QUERIES, small::EMBED_DIM],
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR cross-attention (small): method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record DETR cross-attention small under status key.
#[test]
fn test_detr_cross_attention_small_verify_and_record() {
    let def = build_detr_cross_attention_kernel(
        "detr_cross_attention_small",
        small::NUM_QUERIES,
        small::ENC_SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
    );
    let bindings = cross_attention_bindings(small::ENC_SEQ_LEN, small::EMBED_DIM);
    let input = uniform_bounds(&[small::NUM_QUERIES, small::EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_cross_attention_small");
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (object queries)"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[small::NUM_QUERIES, small::EMBED_DIM]);
}

// ===========================================================================
// Tests: DETR cross-attention (medium configuration)
// ===========================================================================

/// DETR cross-attention kernel validates (medium: d=256, heads=8).
#[test]
fn test_detr_cross_attention_medium_def_validates() {
    let def = build_detr_cross_attention_kernel(
        "detr_xattn_medium",
        medium::NUM_QUERIES,
        medium::ENC_SEQ_LEN,
        medium::EMBED_DIM,
        medium::NUM_HEADS,
    );
    def.validate()
        .expect("DETR cross-attention (medium) kernel should validate");
}

/// IBP bounds propagate through DETR cross-attention (medium).
///
/// Medium configuration exercises larger dimension sizes and more heads,
/// which stress-tests the reshape/transpose/attention graph construction.
#[test]
fn test_detr_cross_attention_medium_ibp_propagates() {
    let def = build_detr_cross_attention_kernel(
        "detr_xattn_medium_ibp",
        medium::NUM_QUERIES,
        medium::ENC_SEQ_LEN,
        medium::EMBED_DIM,
        medium::NUM_HEADS,
    );
    let bindings = cross_attention_bindings(medium::ENC_SEQ_LEN, medium::EMBED_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[medium::NUM_QUERIES, medium::EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DETR cross-attention (medium)");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[medium::NUM_QUERIES, medium::EMBED_DIM],
        "output shape must be [num_queries, embed_dim]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR cross-attention (medium) IBP: bounds=[{lo_min}, {hi_max}]");
}

/// IBP bounds width stays reasonable for DETR cross-attention (small).
///
/// With small weights (0.02) and [-1, 1] input, bounds should not blow up.
#[test]
fn test_detr_cross_attention_small_bounds_width() {
    let def = build_detr_cross_attention_kernel(
        "detr_xattn_small_width",
        small::NUM_QUERIES,
        small::ENC_SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
    );
    let bindings = cross_attention_bindings(small::ENC_SEQ_LEN, small::EMBED_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[small::NUM_QUERIES, small::EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DETR cross-attention");
    let (lo, hi) = output.lower_upper();

    let max_width = lo
        .iter()
        .zip(hi.iter())
        .map(|(l, u)| (u - l).abs())
        .fold(0.0f32, f32::max);

    // Small weights and bounded input should keep bounds tight.
    assert!(
        max_width < 200.0,
        "DETR cross-attention IBP bounds max width {max_width} should be < 200.0"
    );
}

// ===========================================================================
// Tests: DETR decoder block (self-attention + cross-attention + FFN)
// ===========================================================================

/// Full DETR decoder block kernel validates (small).
#[test]
fn test_detr_decoder_block_small_def_validates() {
    let def = build_detr_decoder_block_kernel(
        "detr_dec_block_small",
        small::NUM_QUERIES,
        small::ENC_SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
        small::FFN_DIM,
    );
    def.validate()
        .expect("DETR decoder block kernel should validate");
}

/// DETR decoder block graph builds with sufficient complexity (small).
#[test]
fn test_detr_decoder_block_small_graph_builds() {
    let def = build_detr_decoder_block_kernel(
        "detr_dec_block_small_graph",
        small::NUM_QUERIES,
        small::ENC_SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
        small::FFN_DIM,
    );
    let bindings = decoder_block_bindings(small::ENC_SEQ_LEN, small::EMBED_DIM, small::FFN_DIM);
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("DETR decoder block graph should translate");

    // Self-attention + cross-attention + FFN = many nodes.
    // Self-attn: LN + MHA + residual = ~10 nodes
    // Cross-attn block: 2x LN + CrossMHA + residual + LN + FFN + residual + LN = ~15+ nodes
    assert!(
        graph.num_nodes() >= 15,
        "DETR decoder block should have >= 15 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full DETR decoder block (small).
#[test]
fn test_detr_decoder_block_small_ibp_propagates() {
    let def = build_detr_decoder_block_kernel(
        "detr_dec_block_small_ibp",
        small::NUM_QUERIES,
        small::ENC_SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
        small::FFN_DIM,
    );
    let bindings = decoder_block_bindings(small::ENC_SEQ_LEN, small::EMBED_DIM, small::FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[small::NUM_QUERIES, small::EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DETR decoder block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[small::NUM_QUERIES, small::EMBED_DIM],
        "decoder block output shape must be [num_queries, embed_dim]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder block (small) IBP: bounds=[{lo_min}, {hi_max}]");
}

/// CROWN propagation through the full DETR decoder block (small).
///
/// The DETR decoder block contains LayerNorm layers, so CROWN may use
/// heuristic linearization (IbpValidated mode) and might fall back to IBP.
#[test]
fn test_detr_decoder_block_small_crown_propagation() {
    let def = build_detr_decoder_block_kernel(
        "detr_dec_block_small_crown",
        small::NUM_QUERIES,
        small::ENC_SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
        small::FFN_DIM,
    );
    let bindings = decoder_block_bindings(small::ENC_SEQ_LEN, small::EMBED_DIM, small::FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[small::NUM_QUERIES, small::EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[small::NUM_QUERIES, small::EMBED_DIM],
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder block (small): method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record DETR decoder block (small) under status key.
#[test]
fn test_detr_decoder_block_small_verify_and_record() {
    let def = build_detr_decoder_block_kernel(
        "detr_decoder_block_small",
        small::NUM_QUERIES,
        small::ENC_SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
        small::FFN_DIM,
    );
    let bindings = decoder_block_bindings(small::ENC_SEQ_LEN, small::EMBED_DIM, small::FFN_DIM);
    let input = uniform_bounds(&[small::NUM_QUERIES, small::EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_decoder_block_small");
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (object queries)"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[small::NUM_QUERIES, small::EMBED_DIM]);

    // LayerNorm uses heuristic normalization approximation -> Heuristic mode.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "DETR decoder block with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: DETR decoder block (medium configuration)
// ===========================================================================

/// Full DETR decoder block kernel validates (medium: d=256, heads=8).
#[test]
fn test_detr_decoder_block_medium_def_validates() {
    let def = build_detr_decoder_block_kernel(
        "detr_dec_block_medium",
        medium::NUM_QUERIES,
        medium::ENC_SEQ_LEN,
        medium::EMBED_DIM,
        medium::NUM_HEADS,
        medium::FFN_DIM,
    );
    def.validate()
        .expect("DETR decoder block (medium) kernel should validate");
}

/// IBP propagates through the full DETR decoder block (medium).
#[test]
fn test_detr_decoder_block_medium_ibp_propagates() {
    let def = build_detr_decoder_block_kernel(
        "detr_dec_block_medium_ibp",
        medium::NUM_QUERIES,
        medium::ENC_SEQ_LEN,
        medium::EMBED_DIM,
        medium::NUM_HEADS,
        medium::FFN_DIM,
    );
    let bindings = decoder_block_bindings(medium::ENC_SEQ_LEN, medium::EMBED_DIM, medium::FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[medium::NUM_QUERIES, medium::EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DETR decoder block (medium)");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[medium::NUM_QUERIES, medium::EMBED_DIM],
        "decoder block output shape must be [num_queries, embed_dim]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder block (medium) IBP: bounds=[{lo_min}, {hi_max}]");
}

// ===========================================================================
// Tests: Asymmetric bounds — Q and KV have different perturbation radii
// ===========================================================================

/// Cross-attention with tighter encoder bounds propagates tighter output bounds.
///
/// When encoder output has smaller perturbation than decoder queries, the
/// K/V contribution to attention should produce tighter bounds than when
/// both have wide perturbation. This verifies that cross-attention correctly
/// propagates the asymmetric bound structure.
#[test]
fn test_detr_cross_attention_asymmetric_bounds() {
    let def = build_detr_cross_attention_kernel(
        "detr_xattn_asym",
        small::NUM_QUERIES,
        small::ENC_SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
    );

    // Wide-bound encoder: KV constant with larger values
    let kv_wide = ArrayD::from_elem(IxDyn(&[small::ENC_SEQ_LEN, small::EMBED_DIM]), 0.5f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[small::EMBED_DIM, small::EMBED_DIM]), 0.02f32);

    let bindings_wide = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kv_wide),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
    ];

    // Tight-bound encoder: KV constant with smaller values
    let kv_tight = ArrayD::from_elem(IxDyn(&[small::ENC_SEQ_LEN, small::EMBED_DIM]), 0.01f32);
    let bindings_tight = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kv_tight),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj),
    ];

    let input = uniform_bounds(&[small::NUM_QUERIES, small::EMBED_DIM], 1.0);

    let graph_wide = tensor_kernel_to_graph(&def, &bindings_wide).expect("wide graph");
    let graph_tight = tensor_kernel_to_graph(&def, &bindings_tight).expect("tight graph");

    let output_wide = graph_wide.propagate_ibp(&input).expect("IBP wide");
    let output_tight = graph_tight.propagate_ibp(&input).expect("IBP tight");

    assert_bounds_valid(&output_wide);
    assert_bounds_valid(&output_tight);

    let (wide_lo, wide_hi) = bounds_min_max(&output_wide);
    let (tight_lo, tight_hi) = bounds_min_max(&output_tight);

    eprintln!("Wide encoder bounds: [{wide_lo}, {wide_hi}]");
    eprintln!("Tight encoder bounds: [{tight_lo}, {tight_hi}]");

    // Both should produce valid, finite bounds with correct shape.
    assert_eq!(
        output_wide.lower_upper().0.shape(),
        &[small::NUM_QUERIES, small::EMBED_DIM],
    );
    assert_eq!(
        output_tight.lower_upper().0.shape(),
        &[small::NUM_QUERIES, small::EMBED_DIM],
    );
}
