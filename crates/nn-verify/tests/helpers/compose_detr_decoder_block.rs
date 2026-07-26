// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: DETR decoder full block compose verification.
//!
//! Verifies bounds propagation through each sub-block of a DETR decoder layer
//! and the composed whole:
//!
//! 1. **Self-attention** on object queries (Q=K=V from decoder, bidirectional)
//! 2. **Cross-attention** (Q from decoder, K/V from encoder features as constant)
//! 3. **FFN** (Linear -> ReLU -> Linear)
//! 4. **Full decoder block**: self-attn -> LN -> cross-attn -> LN -> FFN -> LN
//!    with residual connections around each sub-block (pre-norm architecture)
//! 5. **verify_and_assert** recording for status tracking
//!
//! Architecture (Carion et al. 2020, "End-to-End Object Detection with Transformers"):
//!   - Pre-norm: LayerNorm before each sub-block
//!   - Self-attention: object queries attend to each other
//!   - Cross-attention: object queries attend to encoder features
//!   - FFN: two-layer MLP with ReLU activation
//!   - Residual connections around each sub-block
//!
//! Part of #3548: DETR decoder full block compose verification.

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
// Dimensions (per issue #3548)
// ===========================================================================

/// Number of learned object queries (decoder side).
const NUM_QUERIES: usize = 4;
/// Embedding / model dimension.
const EMBED_DIM: usize = 64;
/// Number of attention heads (head_dim = EMBED_DIM / NUM_HEADS = 16).
const NUM_HEADS: usize = 4;
/// FFN intermediate dimension.
const FFN_DIM: usize = 256;
/// Encoder output sequence length (flattened spatial features).
const ENC_SEQ_LEN: usize = 8;
/// Small weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// Builder helpers
// ===========================================================================

/// Build a self-attention kernel on object queries (pre-norm + residual).
///
/// Input: `[NUM_QUERIES, EMBED_DIM]` (Variable -- object queries).
/// Output: `[NUM_QUERIES, EMBED_DIM]`.
///
/// Sub-block 1 of the DETR decoder layer:
///   LayerNorm(x) -> MHA(bidirectional) -> + x (residual)
fn build_self_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_dec_self_attn");

    let input = b.add_input("object_queries", &[NUM_QUERIES, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    let shape = [NUM_QUERIES, EMBED_DIM];

    // Pre-norm: LayerNorm
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);

    // Multi-head self-attention (bidirectional -- queries attend to each other)
    let attn = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid self-attention");

    // Residual connection
    let out = b.add_binary_add(input, attn, &shape);

    b.build(out).expect("valid self-attention kernel")
}

/// Bindings for the self-attention kernel.
fn self_attention_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
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

/// Build a cross-attention kernel (Q from decoder, K/V from encoder).
///
/// Q input: `[NUM_QUERIES, EMBED_DIM]` (Variable -- decoder object queries).
/// KV input: `[ENC_SEQ_LEN, EMBED_DIM]` (Constant -- encoder output).
/// Output: `[NUM_QUERIES, EMBED_DIM]`.
///
/// Sub-block 2 of the DETR decoder layer:
///   LayerNorm(x) -> CrossMHA(x, encoder_output) -> + x (residual)
fn build_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_dec_cross_attn");

    let q_input = b.add_input("object_queries", &[NUM_QUERIES, EMBED_DIM]);
    let kv_input = b.add_input("encoder_output", &[ENC_SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    let shape = [NUM_QUERIES, EMBED_DIM];

    // Pre-norm on Q (decoder) side
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

    b.build(out).expect("valid cross-attention kernel")
}

/// Bindings for the cross-attention kernel.
fn cross_attention_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let kv_const = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, d]), 0.1f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                 // object_queries [Q, D]
        TensorParamBinding::ConstantTensor(kv_const), // encoder_output [S, D]
        TensorParamBinding::ConstantScalar(1e-5),     // eps
        TensorParamBinding::ConstantTensor(ln_w),     // ln_weight [D]
        TensorParamBinding::ConstantTensor(ln_b),     // ln_bias [D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj),   // out_weight [D, D]
    ]
}

/// Build an FFN block (Linear -> ReLU -> Linear) with pre-norm and residual.
///
/// Input: `[NUM_QUERIES, EMBED_DIM]` (Variable).
/// Output: `[NUM_QUERIES, EMBED_DIM]`.
///
/// Sub-block 3 of the DETR decoder layer:
///   LayerNorm(x) -> Linear(D, FFN_DIM) -> ReLU -> Linear(FFN_DIM, D) -> + x
fn build_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_dec_ffn");

    let input = b.add_input("x", &[NUM_QUERIES, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, EMBED_DIM]);
    let ffn2_w = b.add_input("ffn2_weight", &[EMBED_DIM, FFN_DIM]);

    let shape = [NUM_QUERIES, EMBED_DIM];
    let ffn_shape = [NUM_QUERIES, FFN_DIM];

    // Pre-norm: LayerNorm
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);

    // FFN: Linear -> ReLU -> Linear
    let ffn1 = b.add_linear(normed, ffn1_w, None, &ffn_shape);
    let act = b.add_relu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);

    // Residual connection
    let out = b.add_binary_add(input, ffn2, &shape);

    b.build(out).expect("valid FFN kernel")
}

/// Bindings for the FFN kernel.
fn ffn_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    vec![
        TensorParamBinding::Variable,               // x [Q, D]
        TensorParamBinding::ConstantScalar(1e-5),   // eps
        TensorParamBinding::ConstantTensor(ln_w),   // ln_weight [D]
        TensorParamBinding::ConstantTensor(ln_b),   // ln_bias [D]
        TensorParamBinding::ConstantTensor(w_ffn1), // ffn1_weight [FFN_DIM, D]
        TensorParamBinding::ConstantTensor(w_ffn2), // ffn2_weight [D, FFN_DIM]
    ]
}

/// Build a full DETR decoder block: self-attn -> LN -> cross-attn -> LN -> FFN -> LN.
///
/// Input: `[NUM_QUERIES, EMBED_DIM]` (Variable -- object queries).
/// Encoder output: `[ENC_SEQ_LEN, EMBED_DIM]` (Constant).
/// Output: `[NUM_QUERIES, EMBED_DIM]`.
///
/// Pre-norm structure with 3 residual connections:
/// 1. LN -> MHA(bidirectional self-attn) -> + residual
/// 2. LN -> CrossMHA(encoder_output) -> + residual
/// 3. LN -> Linear(D, FFN_DIM) -> ReLU -> Linear(FFN_DIM, D) -> + residual
fn build_full_decoder_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_dec_full_block");

    // Inputs
    let q_input = b.add_input("object_queries", &[NUM_QUERIES, EMBED_DIM]);
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

    let shape = [NUM_QUERIES, EMBED_DIM];
    let ffn_shape = [NUM_QUERIES, FFN_DIM];

    // --- Sub-block 1: Self-attention on object queries ---
    let sa_normed = b.add_layer_norm(q_input, eps, 1, sa_ln_w, sa_ln_b, &shape);
    let sa_out = b
        .add_multi_head_attention(
            sa_normed,
            sa_q_w,
            sa_k_w,
            sa_v_w,
            sa_out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid self-attention");
    let residual1 = b.add_binary_add(q_input, sa_out, &shape);

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

    // --- Sub-block 3: FFN (Linear -> ReLU -> Linear) ---
    let ffn_normed = b.add_layer_norm(residual2, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
    let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
    let act = b.add_relu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(residual2, ffn2, &shape);

    b.build(out).expect("valid full decoder block kernel")
}

/// Bindings for the full decoder block kernel.
fn full_decoder_block_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let kv_const = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, d]), 0.1f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);
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
        // Cross-attention weights
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ca_ln_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ca_ln_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_v_weight
        TensorParamBinding::ConstantTensor(w_proj),       // ca_out_weight
        // FFN weights
        TensorParamBinding::ConstantTensor(ln_w), // ffn_ln_weight
        TensorParamBinding::ConstantTensor(ln_b), // ffn_ln_bias
        TensorParamBinding::ConstantTensor(w_ffn1), // ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2), // ffn2_weight
    ]
}

// ===========================================================================
// Tests: Self-attention on object queries (sub-block 1)
// ===========================================================================

/// Self-attention TensorKernelDef validates.
#[test]
fn test_detr_dec_self_attn_def_validates() {
    let def = build_self_attention_kernel();
    def.validate()
        .expect("self-attention kernel should validate");
}

/// IBP bounds propagate through self-attention on object queries.
///
/// Object queries attend to each other with bidirectional attention.
/// With small weights (0.02) and [-1, 1] input, bounds should remain finite
/// and the residual connection preserves at least the input range.
#[test]
fn test_detr_dec_self_attn_ibp_propagates() {
    let def = build_self_attention_kernel();
    let bindings = self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through self-attention");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, EMBED_DIM],
        "output shape must be [NUM_QUERIES, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder self-attn IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through self-attention on object queries.
///
/// LayerNorm requires heuristic CROWN linearization (IbpValidated mode).
#[test]
fn test_detr_dec_self_attn_crown_propagation() {
    let def = build_self_attention_kernel();
    let bindings = self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder self-attn: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// Tests: Cross-attention (sub-block 2)
// ===========================================================================

/// Cross-attention TensorKernelDef validates.
#[test]
fn test_detr_dec_cross_attn_def_validates() {
    let def = build_cross_attention_kernel();
    def.validate()
        .expect("cross-attention kernel should validate");
}

/// IBP bounds propagate through cross-attention.
///
/// Q comes from Variable decoder queries, K/V from Constant encoder output.
/// Output shape follows Q: [NUM_QUERIES, EMBED_DIM].
#[test]
fn test_detr_dec_cross_attn_ibp_propagates() {
    let def = build_cross_attention_kernel();
    let bindings = cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attention");

    // Output shape matches Q (decoder) sequence length, not KV (encoder).
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, EMBED_DIM],
        "output shape must be [NUM_QUERIES, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder cross-attn IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through cross-attention.
///
/// Cross-attention with constant K/V should allow CROWN to produce tighter
/// bounds since the K/V branch has zero perturbation radius.
#[test]
fn test_detr_dec_cross_attn_crown_propagation() {
    let def = build_cross_attention_kernel();
    let bindings = cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder cross-attn: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// Tests: FFN block (sub-block 3)
// ===========================================================================

/// FFN block TensorKernelDef validates.
#[test]
fn test_detr_dec_ffn_def_validates() {
    let def = build_ffn_kernel();
    def.validate().expect("FFN kernel should validate");
}

/// IBP bounds propagate through the FFN block.
///
/// Linear -> ReLU -> Linear with pre-norm and residual.
/// ReLU clips negative values, which should help keep bounds tighter
/// than an unbounded activation.
#[test]
fn test_detr_dec_ffn_ibp_propagates() {
    let def = build_ffn_kernel();
    let bindings = ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through FFN");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, EMBED_DIM],
        "FFN output shape must be [NUM_QUERIES, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder FFN IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through the FFN block.
///
/// ReLU linearizes cleanly via CROWN (piecewise linear). LayerNorm
/// requires heuristic linearization (IbpValidated mode).
#[test]
fn test_detr_dec_ffn_crown_propagation() {
    let def = build_ffn_kernel();
    let bindings = ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder FFN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// Tests: Full decoder block (all 3 sub-blocks composed)
// ===========================================================================

/// Full decoder block TensorKernelDef validates.
#[test]
fn test_detr_dec_full_block_def_validates() {
    let def = build_full_decoder_block_kernel();
    def.validate()
        .expect("full decoder block kernel should validate");
}

/// Full decoder block translates to NY GraphNetwork with sufficient depth.
#[test]
fn test_detr_dec_full_block_graph_builds() {
    let def = build_full_decoder_block_kernel();
    let bindings = full_decoder_block_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("full decoder block graph should translate");

    // Self-attn: LN + MHA + residual (~12 nodes)
    // Cross-attn: LN + CrossMHA + residual (~12 nodes)
    // FFN: LN + Linear + ReLU + Linear + residual (~6 nodes)
    // Total: at least 20 nodes
    assert!(
        graph.num_nodes() >= 20,
        "full decoder block should have >= 20 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full DETR decoder block.
#[test]
fn test_detr_dec_full_block_ibp_propagates() {
    let def = build_full_decoder_block_kernel();
    let bindings = full_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full decoder block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, EMBED_DIM],
        "decoder block output shape must be [NUM_QUERIES, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder full block IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through the full DETR decoder block.
///
/// The full block has 3 LayerNorms, self-attention softmax, cross-attention
/// softmax, and ReLU -- all requiring CROWN linearization. LayerNorm uses
/// heuristic linearization (IbpValidated mode). CROWN may fall back to IBP
/// due to the depth of chained normalization layers.
#[test]
fn test_detr_dec_full_block_crown_propagation() {
    let def = build_full_decoder_block_kernel();
    let bindings = full_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder full block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// Tests: verify_and_assert recording
// ===========================================================================

/// Verify and record full DETR decoder block under status key.
///
/// Records to `nn_verify_status_detr.json` for ongoing tracking.
#[test]
fn test_detr_dec_full_block_verify_and_record() {
    let def = build_full_decoder_block_kernel();
    let bindings = full_decoder_block_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_decoder_full_block");
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (object queries)"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, EMBED_DIM]);

    // 3 LayerNorms use heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Full decoder block with LayerNorms should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
