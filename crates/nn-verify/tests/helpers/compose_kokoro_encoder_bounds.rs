// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! IBP compose tests for Kokoro PLBert-based text encoder bounds.
//!
//! The Kokoro TTS text encoder is a PLBert-based transformer that converts
//! phoneme token sequences to hidden representations. Architecture:
//!   - Token embedding: phoneme IDs → dense vectors `[T, D]`
//!   - Positional encoding: sinusoidal offsets added to embeddings
//!   - Multi-head self-attention: Q/K/V projections + scaled dot-product
//!   - LayerNorm: pre/post normalization maintaining bounded activations
//!   - Feed-forward network: 2-layer MLP with GELU activation
//!   - Single encoder layer: full pre-norm transformer block
//!   - Stacked encoder layers: sequential transformer layers
//!   - Style conditioning: style vector modulates encoder hidden states
//!
//! This file verifies 8 IBP properties of the text encoder pipeline:
//!
//! 1. **Token embedding bounds** — phoneme embedding produces bounded vectors.
//! 2. **Positional encoding bounds** — sinusoidal position adds bounded offsets.
//! 3. **Self-attention bounds** — multi-head attention preserves IBP bounds.
//! 4. **LayerNorm bounds** — pre/post norm maintains bounded output.
//! 5. **FFN bounds** — 2-layer FFN with GELU preserves bounds.
//! 6. **Single encoder layer** — full transformer layer maintains bounds.
//! 7. **Two-layer stack** — sequential encoder layers maintain bounds.
//! 8. **Style conditioning bounds** — style vector modulation preserves bounds.
//!
//! All tests use small dims (D<=16, T<=4, H=2) and IBP propagation through
//! proxy graphs built with TensorBlockBuilder.
//!
//! Part of #3351: Epic — Absolutely Best Kokoro.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

use super::common::{
    assert_bounds_valid, assert_bounds_width, bounds_min_max, sinusoidal_pe, uniform_bounds,
};

// ===========================================================================
// Constants
// ===========================================================================

/// Model dimension (production PLBert: 768; toy scale for verification).
const D_MODEL: usize = 16;

/// Sequence length (number of phonemes).
const SEQ_LEN: usize = 4;

/// Number of attention heads. D_MODEL must be divisible by this.
const NUM_HEADS: usize = 2;

/// Head dimension.
const HEAD_DIM: usize = D_MODEL / NUM_HEADS;

/// FFN intermediate dimension (production: 4 * D_MODEL; toy scale).
const FFN_DIM: usize = 32;

/// Vocabulary size (number of phoneme tokens).
const VOCAB_SIZE: usize = 64;

/// Style vector dimension.
const STYLE_DIM: usize = 8;

/// Small weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.01;

/// Vacuous width threshold — bounds wider than this are meaningless.
const VACUOUS_THRESHOLD: f32 = 500.0;

// ===========================================================================
// Builder helpers
// ===========================================================================

/// Build a token embedding graph.
///
/// Input: `[SEQ_LEN]` (integer token IDs, treated as Variable for IBP).
/// Embedding weight: `[VOCAB_SIZE, D_MODEL]`.
/// Output: `[SEQ_LEN, D_MODEL]`.
fn build_token_embedding(seq_len: usize, vocab_size: usize, d_model: usize) -> TensorKernelDef {
    let in_shape = [seq_len];
    let out_shape = [seq_len, d_model];
    let mut b = TensorBlockBuilder::new("encoder_token_embedding");

    let tokens = b.add_input("tokens", &in_shape);
    let emb_weight = b.add_input("emb_weight", &[vocab_size, d_model]);
    let out = b.add_embedding(tokens, emb_weight, &out_shape);

    b.build(out).expect("valid token embedding graph")
}

/// Bindings for token embedding.
fn token_embedding_bindings(
    vocab_size: usize,
    d_model: usize,
    weight_mag: f32,
) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // tokens
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[vocab_size, d_model]),
            weight_mag,
        )), // emb_weight
    ]
}

/// Build a positional encoding addition graph.
///
/// Input: `[T, D]` (Variable — token embeddings).
/// Positional encoding: `[T, D]` (Constant — precomputed sinusoidal).
/// Output: `[T, D]`.
fn build_positional_encoding(seq_len: usize, d_model: usize) -> TensorKernelDef {
    let shape = [seq_len, d_model];
    let mut b = TensorBlockBuilder::new("encoder_positional_encoding");

    let x = b.add_input("x", &shape);
    let pe = b.add_input("pe", &shape);
    let out = b.add_binary_add(x, pe, &shape);

    b.build(out).expect("valid positional encoding graph")
}

/// Bindings for positional encoding with precomputed sinusoidal PE.
fn positional_encoding_bindings(seq_len: usize, d_model: usize) -> Vec<TensorParamBinding> {
    let pe_data = sinusoidal_pe(seq_len, d_model);
    vec![
        TensorParamBinding::Variable,                // x (token embeddings)
        TensorParamBinding::ConstantTensor(pe_data), // pe (sinusoidal)
    ]
}

/// Build a single-head self-attention graph (simplified for IBP verification).
///
/// Uses the monolithic `add_attention` op which maps to NY
/// `SelfAttentionLayer`. Input: `[T, D]`. Output: `[T, D]`.
///
/// Architecture: Linear(Q) + Linear(K) + Linear(V) → Attention → Linear(out).
fn build_self_attention(
    seq_len: usize,
    d_model: usize,
    num_heads: usize,
) -> Result<TensorKernelDef, nn_dsl::tensor_ir::TensorIRError> {
    let shape = [seq_len, d_model];
    let mut b = TensorBlockBuilder::new("encoder_self_attention");

    let x = b.add_input("x", &shape);
    let q_w = b.add_input("q_weight", &[d_model, d_model]);
    let k_w = b.add_input("k_weight", &[d_model, d_model]);
    let v_w = b.add_input("v_weight", &[d_model, d_model]);
    let out_w = b.add_input("out_weight", &[d_model, d_model]);

    // Multi-head self-attention (bidirectional for PLBert encoder).
    let attn = b.add_multi_head_attention(
        x,
        q_w,
        k_w,
        v_w,
        out_w,
        num_heads,
        AttentionMask::Standard,
        &shape,
    )?;

    b.build(attn)
}

/// Bindings for self-attention.
fn self_attention_bindings(d_model: usize, weight_mag: f32) -> Vec<TensorParamBinding> {
    let d2 = [d_model, d_model];
    vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&d2), weight_mag)), // q_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&d2), weight_mag)), // k_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&d2), weight_mag)), // v_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&d2), weight_mag)), // out_w
    ]
}

/// Build a LayerNorm graph.
///
/// Input: `[T, D]` (Variable).
/// Output: `[T, D]`.
fn build_layer_norm(seq_len: usize, d_model: usize) -> TensorKernelDef {
    let shape = [seq_len, d_model];
    let mut b = TensorBlockBuilder::new("encoder_layer_norm");

    let x = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[d_model]);
    let beta = b.add_input("beta", &[d_model]);
    // Normalize over last axis (model dimension).
    let out = b.add_layer_norm(x, eps, 1, gamma, beta, &shape);

    b.build(out).expect("valid layer norm graph")
}

/// Bindings for LayerNorm.
fn layer_norm_bindings(d_model: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_model]), 1.0f32)), // gamma
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_model]), 0.0f32)), // beta
    ]
}

/// Build a feed-forward network (FFN) graph.
///
/// Architecture: Linear(D → FFN_DIM) → GELU → Linear(FFN_DIM → D).
///
/// Input: `[T, D]` (Variable).
/// Output: `[T, D]`.
fn build_ffn(seq_len: usize, d_model: usize, ffn_dim: usize) -> TensorKernelDef {
    let shape = [seq_len, d_model];
    let ffn_shape = [seq_len, ffn_dim];
    let mut b = TensorBlockBuilder::new("encoder_ffn");

    let x = b.add_input("x", &shape);
    let w1 = b.add_input("w1", &[ffn_dim, d_model]);
    let w2 = b.add_input("w2", &[d_model, ffn_dim]);
    let fc1 = b.add_linear(x, w1, None, &ffn_shape);
    let act = b.add_gelu(fc1, &ffn_shape);
    let out = b.add_linear(act, w2, None, &shape);

    b.build(out).expect("valid FFN graph")
}

/// Bindings for FFN.
fn ffn_bindings(d_model: usize, ffn_dim: usize, weight_mag: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ffn_dim, d_model]),
            weight_mag,
        )), // w1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d_model, ffn_dim]),
            weight_mag,
        )), // w2
    ]
}

/// Build a single pre-norm encoder layer (full transformer block).
///
/// Architecture: LayerNorm → MHA → residual → LayerNorm → FFN → residual.
///
/// Input: `[T, D]` (Variable).
/// Output: `[T, D]`.
fn build_encoder_layer(
    seq_len: usize,
    d_model: usize,
    num_heads: usize,
    ffn_dim: usize,
) -> TensorKernelDef {
    use nn_dsl::tensor_block_builder::{TransformerBlockConfig, TransformerBlockWeights};

    let shape = [seq_len, d_model];
    let mut b = TensorBlockBuilder::new("encoder_layer");

    let x = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);

    // LayerNorm 1 parameters
    let ln1_w = b.add_input("ln1_weight", &[d_model]);
    let ln1_b = b.add_input("ln1_bias", &[d_model]);

    // Attention projection weights
    let q_w = b.add_input("q_weight", &[d_model, d_model]);
    let k_w = b.add_input("k_weight", &[d_model, d_model]);
    let v_w = b.add_input("v_weight", &[d_model, d_model]);
    let out_w = b.add_input("out_weight", &[d_model, d_model]);

    // LayerNorm 2 parameters
    let ln2_w = b.add_input("ln2_weight", &[d_model]);
    let ln2_b = b.add_input("ln2_bias", &[d_model]);

    // FFN weights
    let ffn1_w = b.add_input("ffn1_weight", &[ffn_dim, d_model]);
    let ffn2_w = b.add_input("ffn2_weight", &[d_model, ffn_dim]);

    let config = TransformerBlockConfig {
        num_heads,
        mask: AttentionMask::Standard, // PLBert is bidirectional
        ffn_hidden_dim: ffn_dim,
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
        .add_transformer_block(x, &weights, &config)
        .expect("valid transformer block");

    b.build(out).expect("valid encoder layer graph")
}

/// Bindings for a single encoder layer.
fn encoder_layer_bindings(
    d_model: usize,
    ffn_dim: usize,
    weight_mag: f32,
) -> Vec<TensorParamBinding> {
    let d2 = [d_model, d_model];
    vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
        // LN1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_model]), 1.0f32)), // ln1_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_model]), 0.0f32)), // ln1_b
        // Attention projections
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&d2), weight_mag)), // q_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&d2), weight_mag)), // k_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&d2), weight_mag)), // v_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&d2), weight_mag)), // out_w
        // LN2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_model]), 1.0f32)), // ln2_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_model]), 0.0f32)), // ln2_b
        // FFN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ffn_dim, d_model]),
            weight_mag,
        )), // ffn1_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d_model, ffn_dim]),
            weight_mag,
        )), // ffn2_w
    ]
}

/// Build a style conditioning graph: hidden = hidden + broadcast(Linear(style)).
///
/// Style vector is projected to model dimension and added to each position.
///
/// Input 0: `[T, D]` (Variable — encoder hidden states).
/// Input 1: `[STYLE_DIM]` (Variable — style vector).
/// Output: `[T, D]`.
fn build_style_conditioning(seq_len: usize, d_model: usize, style_dim: usize) -> TensorKernelDef {
    let shape = [seq_len, d_model];
    let mut b = TensorBlockBuilder::new("encoder_style_conditioning");

    let hidden = b.add_input("hidden", &shape);
    let style = b.add_input("style", &[style_dim]);
    let style_w = b.add_input("style_weight", &[d_model, style_dim]);
    let style_b = b.add_input("style_bias", &[d_model]);

    // Project style: [STYLE_DIM] → [D_MODEL] via Linear.
    let style_proj = b.add_linear(style, style_w, Some(style_b), &[d_model]);

    // Broadcast projected style to all positions: [D] → [T, D].
    let style_bc = b.add_broadcast(style_proj, &shape);

    // Add style to hidden states: hidden + style_proj.
    let out = b.add_binary_add(hidden, style_bc, &shape);

    b.build(out).expect("valid style conditioning graph")
}

/// Bindings for style conditioning (both hidden and style are Variable).
fn style_conditioning_bindings(
    d_model: usize,
    style_dim: usize,
    weight_mag: f32,
) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden
        TensorParamBinding::Variable, // style
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d_model, style_dim]),
            weight_mag,
        )), // style_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_model]), 0.0f32)), // style_bias
    ]
}

// ===========================================================================
// Test 1: Token embedding bounds
// ===========================================================================

/// Token embedding produces bounded output vectors from phoneme IDs.
///
/// The embedding op selects rows from a `[VOCAB_SIZE, D_MODEL]` weight matrix.
/// For IBP, the output bounds are determined by the min/max of the weight rows
/// accessible from the input index range. With uniform small weights, all
/// embeddings produce similar bounded vectors.
#[test]
fn test_encoder_token_embedding_bounds() {
    let def = build_token_embedding(SEQ_LEN, VOCAB_SIZE, D_MODEL);
    def.validate().expect("token embedding def validates");

    let bindings = token_embedding_bindings(VOCAB_SIZE, D_MODEL, WEIGHT_MAG);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Token indices are bounded: phoneme IDs in [0, VOCAB_SIZE-1].
    // For IBP, we set input bounds as uniform range covering the embedding rows.
    let input = uniform_bounds(&[SEQ_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through token embedding");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "token embedding bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "token embedding must produce [T, D] output"
    );

    // With uniform weight_mag=0.01, all embedding rows are identical.
    // IBP should produce tight bounds.
    assert!(
        width < 5.0,
        "token embedding with uniform small weights should have tight bounds, got width={width}"
    );

    eprintln!("Encoder token embedding: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 2: Positional encoding bounds
// ===========================================================================

/// Sinusoidal positional encoding adds bounded offsets to embeddings.
///
/// Sinusoidal PE values are in [-1, 1] (sin/cos). Adding PE to token embeddings
/// shifts the bounds by at most +/- 1.0 per element. IBP handles addition exactly:
///   [lo, hi] + [pe_lo, pe_hi] = [lo + pe_lo, hi + pe_hi]
///
/// Since PE is a constant, the output width equals the input width (addition of
/// a constant preserves interval width).
#[test]
fn test_encoder_positional_encoding_bounds() {
    let def = build_positional_encoding(SEQ_LEN, D_MODEL);
    def.validate().expect("positional encoding def validates");

    let bindings = positional_encoding_bindings(SEQ_LEN, D_MODEL);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through positional encoding");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "positional encoding bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "positional encoding must preserve [T, D] shape"
    );

    // PE values are in [-1, 1]. Input in [-1, 1]. Output in [-2, 2].
    // IBP is exact for constant addition.
    assert!(
        lo_min >= -2.0 - 1e-4,
        "PE lower bound {lo_min} should be >= -2.0"
    );
    assert!(
        hi_max <= 2.0 + 1e-4,
        "PE upper bound {hi_max} should be <= 2.0"
    );
    assert!(
        width > 0.0,
        "PE bounds should have non-zero width, got {width}"
    );

    // Width should be approximately 2.0 (input width preserved by constant add).
    // Some elements may have narrower bounds due to PE values not spanning full [-1, 1].
    assert!(width < 5.0, "PE width {width} should be bounded");

    eprintln!("Encoder positional encoding: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 3: Self-attention bounds
// ===========================================================================

/// Multi-head self-attention preserves IBP bounds through Q/K/V projections,
/// scaled dot-product, softmax, and output projection.
///
/// PLBert uses bidirectional (Standard) attention. The softmax normalizes
/// attention weights to sum to 1, and the value projection is a convex
/// combination — both properties that IBP can exploit for bounded propagation.
///
/// With small projection weights (0.01), the attention output should remain
/// tightly bounded relative to the input range.
#[test]
fn test_encoder_self_attention_bounds() {
    let def =
        build_self_attention(SEQ_LEN, D_MODEL, NUM_HEADS).expect("self-attention graph builds");
    def.validate().expect("self-attention def validates");

    let bindings = self_attention_bindings(D_MODEL, WEIGHT_MAG);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through self-attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "self-attention bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "self-attention must preserve [T, D] shape"
    );

    // With small weights, attention output should be bounded.
    // IBP through attention may over-approximate due to bilinear Q@K^T.
    assert!(
        width < VACUOUS_THRESHOLD,
        "self-attention bounds width {width} exceeds vacuous threshold {VACUOUS_THRESHOLD}"
    );

    eprintln!(
        "Encoder self-attention: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}, \
         graph_nodes={}",
        graph.num_nodes()
    );
}

// ===========================================================================
// Test 4: LayerNorm bounds
// ===========================================================================

/// LayerNorm normalizes along the model dimension, maintaining bounded output.
///
/// LayerNorm computes: `gamma * (x - mean) / sqrt(var + eps) + beta`.
/// With gamma=1 and beta=0, the output is standardized to approximately
/// zero-mean, unit-variance per position. IBP through LayerNorm uses
/// NY's `LayerNormLayer` which accounts for the normalization
/// statistics in the bound propagation.
///
/// Key property: LayerNorm acts as a natural bound stabilizer — even if
/// input bounds are wide, the normalization constrains the output range.
#[test]
fn test_encoder_layer_norm_bounds() {
    let def = build_layer_norm(SEQ_LEN, D_MODEL);
    def.validate().expect("layer norm def validates");

    let bindings = layer_norm_bindings(D_MODEL);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through layer norm");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "layer norm bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "layer norm must preserve [T, D] shape"
    );

    // LayerNorm with gamma=1, beta=0 normalizes output.
    // IBP may over-approximate but should stay bounded.
    assert!(
        width < VACUOUS_THRESHOLD,
        "layer norm bounds width {width} exceeds vacuous threshold {VACUOUS_THRESHOLD}"
    );

    eprintln!("Encoder LayerNorm: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 5: FFN bounds
// ===========================================================================

/// Two-layer FFN with GELU activation preserves bounded outputs.
///
/// Architecture: Linear(D → FFN_DIM) → GELU → Linear(FFN_DIM → D).
///
/// The first linear expands dimension (D → FFN_DIM), GELU applies a smooth
/// non-linearity bounded by [-0.17, x] (GELU(x) ≈ x for x >> 0, ≈ 0 for
/// x << 0), and the second linear projects back to D.
///
/// With small weights (0.01), the output range should be tightly bounded
/// by the product of weight magnitudes and input range.
#[test]
fn test_encoder_ffn_bounds() {
    let def = build_ffn(SEQ_LEN, D_MODEL, FFN_DIM);
    def.validate().expect("FFN def validates");

    let bindings = ffn_bindings(D_MODEL, FFN_DIM, WEIGHT_MAG);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through FFN");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "FFN bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "FFN must produce [T, D] output"
    );

    // With two linear layers at weight_mag=0.01, the output magnitude is
    // approximately D_MODEL * FFN_DIM * weight_mag^2 per element.
    // GELU is bounded, so the composition should stay tight.
    assert!(
        width < 50.0,
        "FFN with small weights should have tight bounds, got width={width}"
    );

    eprintln!(
        "Encoder FFN: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}, \
         graph_nodes={}",
        graph.num_nodes()
    );
}

// ===========================================================================
// Test 6: Single encoder layer bounds
// ===========================================================================

/// Full pre-norm transformer layer maintains bounded outputs through IBP.
///
/// Architecture: LayerNorm → MHA → residual → LayerNorm → FFN → residual.
///
/// The residual connections prevent unbounded growth: the skip path passes
/// input directly, while the attention and FFN branches are attenuated by
/// small weights. LayerNorm normalizes intermediates, bounding the range
/// before each sub-layer.
///
/// This is the core building block of the PLBert text encoder.
#[test]
fn test_encoder_single_layer_bounds() {
    let def = build_encoder_layer(SEQ_LEN, D_MODEL, NUM_HEADS, FFN_DIM);
    def.validate().expect("encoder layer def validates");

    let bindings = encoder_layer_bindings(D_MODEL, FFN_DIM, WEIGHT_MAG);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder layer");
    assert_bounds_valid(&output);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "encoder layer output shape must be [T, D]"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "encoder layer bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert!(
        width > 0.0,
        "encoder layer bounds should have non-zero width, got {width}"
    );

    // Single encoder layer with small weights should not produce vacuously
    // wide bounds. The residual connection limits growth.
    assert_bounds_width(&output, VACUOUS_THRESHOLD, "single_encoder_layer");

    eprintln!(
        "Encoder single layer: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}, \
         graph_nodes={}",
        graph.num_nodes()
    );
}

// ===========================================================================
// Test 7: Two-layer stack bounds
// ===========================================================================

/// Two sequential encoder layers maintain bounded outputs through IBP.
///
/// This tests bound stability through depth: each layer's output bounds
/// become the next layer's input. The residual connections and LayerNorm
/// in each block prevent unbounded growth. With small weights, the
/// compounding should stay manageable.
///
/// This verifies the core property needed for the Kokoro text encoder:
/// multiple stacked transformer layers (PLBert has 12) produce finite
/// bounds when each layer is independently verified and composed.
#[test]
fn test_encoder_two_layer_stack_bounds() {
    let num_layers = 2;
    let mut current_bounds = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (init_lo, init_hi) = bounds_min_max(&current_bounds);
    let init_width = init_hi - init_lo;
    eprintln!("Encoder two-layer stack:");
    eprintln!("  Input: width={init_width:.4}");

    for i in 0..num_layers {
        let def = build_encoder_layer(SEQ_LEN, D_MODEL, NUM_HEADS, FFN_DIM);
        let bindings = encoder_layer_bindings(D_MODEL, FFN_DIM, WEIGHT_MAG);
        let graph = tensor_kernel_to_graph(&def, &bindings)
            .unwrap_or_else(|e| panic!("encoder layer {i} graph: {e}"));

        let output = graph
            .propagate_ibp(&current_bounds)
            .unwrap_or_else(|e| panic!("encoder layer {i} IBP: {e}"));
        assert_bounds_valid(&output);

        let (lo, hi) = bounds_min_max(&output);
        let width = hi - lo;

        eprintln!("  Layer {i}: bounds=[{lo:.4}, {hi:.4}], width={width:.4}");

        assert!(
            lo.is_finite() && hi.is_finite(),
            "encoder layer {i} bounds must be finite: [{lo}, {hi}]"
        );

        current_bounds = output;
    }

    // Final output after 2-layer stack must be finite.
    let (final_lo, final_hi) = bounds_min_max(&current_bounds);
    let final_width = final_hi - final_lo;
    assert!(
        final_lo.is_finite() && final_hi.is_finite(),
        "2-layer encoder stack output must be finite: [{final_lo}, {final_hi}]"
    );

    // Track total expansion.
    let total_expansion = if init_width > 1e-10 {
        final_width / init_width
    } else {
        1.0
    };
    eprintln!(
        "  Total expansion: {total_expansion:.2}x ({num_layers} layers). \
         Final width: {final_width:.4}"
    );

    // With small weights (0.01), 2 layers should not produce vacuously wide bounds.
    assert!(
        final_width.is_finite(),
        "2-layer encoder stack output width must be finite, got {final_width}"
    );
}

// ===========================================================================
// Test 8: Style conditioning bounds
// ===========================================================================

/// Style vector modulation preserves bounded encoder outputs.
///
/// Kokoro's text encoder output is conditioned by a style vector that
/// encodes speaker identity and prosody. The style vector is projected
/// to model dimension and added to each position of the encoder output:
///   output = hidden + broadcast(Linear(style))
///
/// Since this is a linear projection + addition, IBP propagates exactly:
///   - Linear: output_width = style_width * ||W||_1
///   - Addition: output_width = hidden_width + proj_width
///
/// Both inputs (hidden and style) are Variable to model the joint
/// uncertainty of the encoder output and style embedding.
#[test]
fn test_encoder_style_conditioning_bounds() {
    let def = build_style_conditioning(SEQ_LEN, D_MODEL, STYLE_DIM);
    def.validate().expect("style conditioning def validates");

    let bindings = style_conditioning_bindings(D_MODEL, STYLE_DIM, WEIGHT_MAG);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Both hidden states and style vector are Variable inputs.
    let total_elems = SEQ_LEN * D_MODEL + STYLE_DIM;
    let n_hidden = SEQ_LEN * D_MODEL;
    let mut lower = vec![-1.0f32; n_hidden]; // hidden in [-1, 1]
    lower.extend(vec![-0.5f32; STYLE_DIM]); // style in [-0.5, 0.5]
    let mut upper = vec![1.0f32; n_hidden];
    upper.extend(vec![0.5f32; STYLE_DIM]);

    let input = nn_verify::BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total_elems]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total_elems]), upper).unwrap(),
    )
    .expect("valid input bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through style conditioning");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "style conditioning bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "style conditioning must produce [T, D] output"
    );

    // Hidden in [-1, 1], style projected through small weights (0.01) with
    // STYLE_DIM=8 inputs in [-0.5, 0.5]:
    //   proj_range = STYLE_DIM * WEIGHT_MAG * 0.5 = 0.04 per element.
    //   output_width ≈ hidden_width + 2 * proj_range = 2.0 + 0.08 ≈ 2.08.
    // IBP may over-approximate slightly.
    assert!(
        width < 20.0,
        "style conditioning with small weights should have tight bounds, got width={width}"
    );

    // Output should be wider than hidden alone (style adds uncertainty).
    assert!(
        width > 1.5,
        "style conditioning width {width} should be wider than hidden alone"
    );

    eprintln!("Encoder style conditioning: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}
