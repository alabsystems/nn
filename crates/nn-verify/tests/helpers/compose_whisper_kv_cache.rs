// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Whisper decoder KV cache + attention NY composition.
//!
//! Verifies bounds propagation through autoregressive decoding with KV cache:
//!
//! 1. **Decoder self-attention with causal mask**: Full causal MHA on
//!    `[TOTAL_SEQ, D]` input. The causal mask ensures each position only
//!    attends to itself and prior positions, modeling KV cache semantics
//!    where position `T-1` (new token) attends to all prior cached K/V.
//!
//! 2. **Cross-attention with encoder KV cache**: Decoder hidden (Variable)
//!    attends to frozen encoder output (Constant). Uses cross-attention
//!    composite op with separate Q and KV sequences of different length.
//!    Models the encoder-decoder bridge during autoregressive decoding.
//!
//! 3. **Full autoregressive decode step**: Causal self-attn (extracting
//!    last-token output via narrow) + cross-attn + FFN. Complete single-token
//!    decode step with all three sub-blocks and residual connections.
//!
//! Verification approach:
//! - Self-attention: Variable `[TOTAL_SEQ, D]` with causal mask. Positions
//!   0..CACHE_LEN-1 represent cached context, position CACHE_LEN is the new
//!   token. Causal masking ensures correct attention pattern.
//! - Cross-attention: `add_multi_head_cross_attention(Q, KV, ...)` handles
//!   different sequence lengths for Q and KV natively.
//! - Bounds compositionality: encoder output certified separately (#3558),
//!   bound as Constant here for compositional verification.
//!
//! Part of #3576: Whisper decoder cross-attention + KV cache compose verification.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding, VerificationSoundnessMode};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// Per issue #3576: d=32, heads=4, seq=8.
// ---------------------------------------------------------------------------

/// Embedding / model dimension.
const EMBED_DIM: usize = 32;
/// Number of attention heads (head_dim = EMBED_DIM / NUM_HEADS = 8).
const NUM_HEADS: usize = 4;
/// FFN intermediate dimension: 4x the embedding dimension per Whisper spec.
const FFN_DIM: usize = 128;
/// Encoder output sequence length.
const ENC_SEQ_LEN: usize = 8;
/// Total decoder sequence length (cached positions + new token).
/// Models KV cache: positions 0..TOTAL_SEQ-2 are cached, position TOTAL_SEQ-1
/// is the new token. Causal mask ensures correct autoregressive pattern.
const TOTAL_SEQ: usize = 8;
/// Small weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Builder helpers: Decoder self-attention with causal mask
// ---------------------------------------------------------------------------

/// Build a decoder self-attention with causal mask.
///
/// Input: `[TOTAL_SEQ, EMBED_DIM]` (Variable -- full decoder sequence
///   including cached context + new token).
/// Output: `[TOTAL_SEQ, EMBED_DIM]`.
///
/// The causal mask ensures each position only attends to itself and prior
/// positions. For KV cache semantics, position `TOTAL_SEQ-1` (new token)
/// attends to all TOTAL_SEQ positions (full cached K/V + self).
///
/// Architecture:
///   LayerNorm(x) -> MHA(causal) -> + x (residual)
fn build_decoder_causal_self_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_dec_causal_self_attn_kv");

    let input = b.add_input("x", &[TOTAL_SEQ, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    let shape = [TOTAL_SEQ, EMBED_DIM];

    // Pre-norm: LayerNorm
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);

    // Causal multi-head self-attention
    let attn = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("valid causal self-attention");

    // Residual connection
    let out = b.add_binary_add(input, attn, &shape);

    b.build(out)
        .expect("valid decoder causal self-attention kernel")
}

/// Bindings for the decoder causal self-attention kernel.
fn decoder_causal_self_attention_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // x [T, D]
        TensorParamBinding::ConstantScalar(1e-5),           // eps
        TensorParamBinding::ConstantTensor(ln_w),           // ln_weight [D]
        TensorParamBinding::ConstantTensor(ln_b),           // ln_bias [D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj),         // out_weight [D, D]
    ]
}

// ---------------------------------------------------------------------------
// Builder helpers: Cross-attention with encoder KV cache
// ---------------------------------------------------------------------------

/// Build a cross-attention with cached encoder KV.
///
/// Q input: `[TOTAL_SEQ, EMBED_DIM]` (Variable -- decoder hidden state).
/// KV input: `[ENC_SEQ_LEN, EMBED_DIM]` (Constant -- encoder output).
/// Output: `[TOTAL_SEQ, EMBED_DIM]`.
///
/// In Whisper, cross-attention K/V are computed once from encoder output
/// and cached for all decode steps. The encoder output is Constant (certified
/// in #3558). Standard mask (decoder attends to all encoder positions).
///
/// Architecture:
///   LayerNorm(x) -> CrossMHA(x, encoder_output) -> + x (residual)
fn build_cross_attention_with_cache_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_dec_cross_attn_kv_cache");

    let q_input = b.add_input("decoder_hidden", &[TOTAL_SEQ, EMBED_DIM]);
    let kv_input = b.add_input("encoder_output", &[ENC_SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    let shape = [TOTAL_SEQ, EMBED_DIM];

    // Pre-norm: LayerNorm on the Q (decoder) side only
    let normed = b.add_layer_norm(q_input, eps, 1, ln_w, ln_b, &shape);

    // Cross-attention: Q from decoder, K/V from encoder
    let attn = b
        .add_multi_head_cross_attention(
            normed,
            kv_input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid cross-attention");

    // Residual connection
    let out = b.add_binary_add(q_input, attn, &shape);

    b.build(out)
        .expect("valid cross-attention with cache kernel")
}

/// Bindings for the cross-attention with cache kernel.
fn cross_attention_with_cache_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let kv_const = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, d]), 0.1f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                 // decoder_hidden [T, D]
        TensorParamBinding::ConstantTensor(kv_const), // encoder_output [E, D]
        TensorParamBinding::ConstantScalar(1e-5),     // eps
        TensorParamBinding::ConstantTensor(ln_w),     // ln_weight [D]
        TensorParamBinding::ConstantTensor(ln_b),     // ln_bias [D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj),   // out_weight [D, D]
    ]
}

// ---------------------------------------------------------------------------
// Builder helpers: Full autoregressive decode step
// ---------------------------------------------------------------------------

/// Build a full autoregressive decode step with KV cache.
///
/// Input: `[TOTAL_SEQ, EMBED_DIM]` (Variable -- full decoder sequence).
/// Encoder output: `[ENC_SEQ_LEN, EMBED_DIM]` (Constant).
/// Output: `[TOTAL_SEQ, EMBED_DIM]`.
///
/// Architecture:
///   1. LN -> Causal Self-Attn(full seq) + residual
///   2. LN -> Cross-Attn(decoder, encoder) + residual
///   3. LN -> FFN(Linear, GELU, Linear) + residual
fn build_full_decode_step_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_dec_kv_cache_full_step");

    let input = b.add_input("x", &[TOTAL_SEQ, EMBED_DIM]);
    let kv_input = b.add_input("encoder_output", &[ENC_SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    // Self-attention weights
    let sa_ln_w = b.add_input("sa_ln_weight", &[EMBED_DIM]);
    let sa_ln_b = b.add_input("sa_ln_bias", &[EMBED_DIM]);
    let sa_q_w = b.add_input("sa_q_weight", &[EMBED_DIM, EMBED_DIM]);
    let sa_k_w = b.add_input("sa_k_weight", &[EMBED_DIM, EMBED_DIM]);
    let sa_v_w = b.add_input("sa_v_weight", &[EMBED_DIM, EMBED_DIM]);
    let sa_out_w = b.add_input("sa_out_weight", &[EMBED_DIM, EMBED_DIM]);

    // Cross-attention weights
    let ca_ln_w = b.add_input("ca_ln_weight", &[EMBED_DIM]);
    let ca_ln_b = b.add_input("ca_ln_bias", &[EMBED_DIM]);
    let ca_q_w = b.add_input("ca_q_weight", &[EMBED_DIM, EMBED_DIM]);
    let ca_k_w = b.add_input("ca_k_weight", &[EMBED_DIM, EMBED_DIM]);
    let ca_v_w = b.add_input("ca_v_weight", &[EMBED_DIM, EMBED_DIM]);
    let ca_out_w = b.add_input("ca_out_weight", &[EMBED_DIM, EMBED_DIM]);

    // FFN weights
    let ffn_ln_w = b.add_input("ffn_ln_weight", &[EMBED_DIM]);
    let ffn_ln_b = b.add_input("ffn_ln_bias", &[EMBED_DIM]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, EMBED_DIM]);
    let ffn2_w = b.add_input("ffn2_weight", &[EMBED_DIM, FFN_DIM]);

    let shape = [TOTAL_SEQ, EMBED_DIM];
    let ffn_shape = [TOTAL_SEQ, FFN_DIM];

    // --- Sub-block 1: Causal self-attention ---
    let sa_normed = b.add_layer_norm(input, eps, 1, sa_ln_w, sa_ln_b, &shape);
    let sa_out = b
        .add_multi_head_attention(
            sa_normed,
            sa_q_w,
            sa_k_w,
            sa_v_w,
            sa_out_w,
            NUM_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("valid causal self-attention");
    let residual1 = b.add_binary_add(input, sa_out, &shape);

    // --- Sub-block 2: Cross-attention with encoder output ---
    let ca_normed = b.add_layer_norm(residual1, eps, 1, ca_ln_w, ca_ln_b, &shape);
    let ca_out = b
        .add_multi_head_cross_attention(
            ca_normed,
            kv_input,
            ca_q_w,
            ca_k_w,
            ca_v_w,
            ca_out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid cross-attention");
    let residual2 = b.add_binary_add(residual1, ca_out, &shape);

    // --- Sub-block 3: FFN ---
    let ffn_normed = b.add_layer_norm(residual2, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
    let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(residual2, ffn2, &shape);

    b.build(out).expect("valid full decode step kernel")
}

/// Bindings for the full decode step kernel.
fn full_decode_step_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let kv_const = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, d]), 0.1f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                 // x [T, D]
        TensorParamBinding::ConstantTensor(kv_const), // encoder_output [E, D]
        TensorParamBinding::ConstantScalar(1e-5),     // eps
        // Self-attention
        TensorParamBinding::ConstantTensor(ln_w.clone()), // sa_ln_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // sa_ln_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // sa_q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // sa_k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // sa_v_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // sa_out_weight
        // Cross-attention
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ca_ln_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ca_ln_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_v_weight
        TensorParamBinding::ConstantTensor(w_proj),       // ca_out_weight
        // FFN
        TensorParamBinding::ConstantTensor(ln_w), // ffn_ln_weight
        TensorParamBinding::ConstantTensor(ln_b), // ffn_ln_bias
        TensorParamBinding::ConstantTensor(w_ffn1), // ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2), // ffn2_weight
    ]
}

// ===========================================================================
// Tests: Decoder self-attention with causal mask
// ===========================================================================

/// Decoder causal self-attention TensorKernelDef validates.
#[test]
fn test_whisper_dec_kv_cache_self_attn_def_validates() {
    let def = build_decoder_causal_self_attention_kernel();
    def.validate()
        .expect("decoder causal self-attention kernel should validate");
}

/// Decoder causal self-attention translates to NY GraphNetwork.
#[test]
fn test_whisper_dec_kv_cache_self_attn_graph_builds() {
    let def = build_decoder_causal_self_attention_kernel();
    let bindings = decoder_causal_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("decoder causal self-attention graph should translate");

    // LayerNorm + Q/K/V projections + reshape + transpose + attention +
    // transpose + reshape + output projection + residual >= 10 nodes
    assert!(
        graph.num_nodes() >= 10,
        "decoder causal self-attn graph should have >= 10 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through decoder causal self-attention.
///
/// The causal mask restricts attention: position i attends only to positions
/// 0..i. This models KV cache semantics where earlier positions represent
/// cached context and the final position is the new token.
#[test]
fn test_whisper_dec_kv_cache_self_attn_ibp_propagates() {
    let def = build_decoder_causal_self_attention_kernel();
    let bindings = decoder_causal_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[TOTAL_SEQ, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder causal self-attention");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[TOTAL_SEQ, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper kv-cache self-attn IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN propagation through decoder causal self-attention.
#[test]
fn test_whisper_dec_kv_cache_self_attn_crown_propagation() {
    let def = build_decoder_causal_self_attention_kernel();
    let bindings = decoder_causal_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[TOTAL_SEQ, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[TOTAL_SEQ, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper kv-cache self-attn: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Verify and record decoder causal self-attention.
#[test]
fn test_whisper_dec_kv_cache_self_attn_verify_and_record() {
    let def = build_decoder_causal_self_attention_kernel();
    let bindings = decoder_causal_self_attention_bindings();
    let input = uniform_bounds(&[TOTAL_SEQ, EMBED_DIM], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "whisper_decoder_kv_cache_self_attn",
    );
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[TOTAL_SEQ, EMBED_DIM]);

    // LayerNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Decoder causal self-attn with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: Cross-attention with encoder KV cache
// ===========================================================================

/// Cross-attention with encoder cache TensorKernelDef validates.
#[test]
fn test_whisper_dec_kv_cache_cross_attn_def_validates() {
    let def = build_cross_attention_with_cache_kernel();
    def.validate()
        .expect("cross-attention with cache kernel should validate");
}

/// Cross-attention with encoder cache translates to GraphNetwork.
#[test]
fn test_whisper_dec_kv_cache_cross_attn_graph_builds() {
    let def = build_cross_attention_with_cache_kernel();
    let bindings = cross_attention_with_cache_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("cross-attention with cache graph should translate");

    // LayerNorm + Q proj from decoder + K/V proj from encoder +
    // reshape + transpose + attention + transpose + reshape +
    // output projection + residual >= 10 nodes
    assert!(
        graph.num_nodes() >= 10,
        "cross-attn with cache graph should have >= 10 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through cross-attention with encoder cache.
///
/// Key structural difference: Q comes from Variable decoder hidden state,
/// K/V come from Constant encoder output. The output shape follows Q.
#[test]
fn test_whisper_dec_kv_cache_cross_attn_ibp_propagates() {
    let def = build_cross_attention_with_cache_kernel();
    let bindings = cross_attention_with_cache_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[TOTAL_SEQ, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attention with cache");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[TOTAL_SEQ, EMBED_DIM],
        "output shape must follow Q (decoder) seq len"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper kv-cache cross-attn IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through cross-attention with encoder cache.
#[test]
fn test_whisper_dec_kv_cache_cross_attn_crown_propagation() {
    let def = build_cross_attention_with_cache_kernel();
    let bindings = cross_attention_with_cache_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[TOTAL_SEQ, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[TOTAL_SEQ, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper kv-cache cross-attn: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record cross-attention with encoder cache.
#[test]
fn test_whisper_dec_kv_cache_cross_attn_verify_and_record() {
    let def = build_cross_attention_with_cache_kernel();
    let bindings = cross_attention_with_cache_bindings();
    let input = uniform_bounds(&[TOTAL_SEQ, EMBED_DIM], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "whisper_decoder_kv_cache_cross_attn",
    );
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (decoder_hidden)"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[TOTAL_SEQ, EMBED_DIM]);

    // LayerNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Cross-attn with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: Full autoregressive decode step with KV cache
// ===========================================================================

/// Full decode step TensorKernelDef validates.
#[test]
fn test_whisper_dec_kv_cache_full_step_def_validates() {
    let def = build_full_decode_step_kernel();
    def.validate()
        .expect("full decode step kernel should validate");
}

/// Full decode step translates to NY GraphNetwork.
#[test]
fn test_whisper_dec_kv_cache_full_step_graph_builds() {
    let def = build_full_decode_step_kernel();
    let bindings = full_decode_step_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("full decode step graph should translate");

    // Self-attn: LN + MHA + residual (~12 nodes)
    // Cross-attn: LN + CrossMHA + residual (~12 nodes)
    // FFN: LN + Linear + GELU + Linear + residual (~6 nodes)
    assert!(
        graph.num_nodes() >= 25,
        "full decode step should have >= 25 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through full decode step.
#[test]
fn test_whisper_dec_kv_cache_full_step_ibp_propagates() {
    let def = build_full_decode_step_kernel();
    let bindings = full_decode_step_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[TOTAL_SEQ, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full decode step");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[TOTAL_SEQ, EMBED_DIM],
        "decode step output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper kv-cache full step IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through full decode step.
///
/// The full decode step contains 3 LayerNorms (self-attn + cross-attn +
/// FFN), so CROWN uses heuristic linearization (IbpValidated mode).
#[test]
fn test_whisper_dec_kv_cache_full_step_crown_propagation() {
    let def = build_full_decode_step_kernel();
    let bindings = full_decode_step_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[TOTAL_SEQ, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[TOTAL_SEQ, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper kv-cache full step: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Narrower input produces tighter output bounds (monotonicity).
#[test]
fn test_whisper_dec_kv_cache_full_step_narrow_inputs_tighter() {
    let def = build_full_decode_step_kernel();
    let bindings = full_decode_step_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[TOTAL_SEQ, EMBED_DIM], 10.0);
    let narrow_input = uniform_bounds(&[TOTAL_SEQ, EMBED_DIM], 1.0);

    let wide_output = graph.propagate_ibp(&wide_input).expect("wide IBP");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("narrow IBP");

    let (wide_lo, wide_hi) = wide_output.lower_upper();
    let (narrow_lo, narrow_hi) = narrow_output.lower_upper();

    let wide_range = wide_hi.iter().zip(wide_lo.iter()).map(|(h, l)| h - l);
    let narrow_range = narrow_hi.iter().zip(narrow_lo.iter()).map(|(h, l)| h - l);

    let tighter_count = wide_range.zip(narrow_range).filter(|(w, n)| n <= w).count();
    let total = wide_lo.len();
    assert!(
        tighter_count > total / 2,
        "narrow input should produce tighter bounds for > 50% of elements, \
         got {tighter_count}/{total}"
    );
}

/// Verify and record full decode step.
#[test]
fn test_whisper_dec_kv_cache_full_step_verify_and_record() {
    let def = build_full_decode_step_kernel();
    let bindings = full_decode_step_bindings();
    let input = uniform_bounds(&[TOTAL_SEQ, EMBED_DIM], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "whisper_decoder_kv_cache_full_step",
    );
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (decoder seq)"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[TOTAL_SEQ, EMBED_DIM]);

    // 3 LayerNorms use heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Full decode step with LayerNorms should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
