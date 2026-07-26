// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Whisper decoder NY composition.
//!
//! Verifies bounds propagation through the three key Whisper decoder sub-blocks:
//!
//! 1. **Causal self-attention** with mask (decoder self-attn): Q=K=V from
//!    decoder hidden state, with causal mask preventing future positions.
//!
//! 2. **Cross-attention**: Q from decoder, K/V from encoder output bound as
//!    constant. This is the encoder-decoder bridge in Whisper.
//!
//! 3. **Full decoder block**: SelfAttn -> LayerNorm -> CrossAttn -> LayerNorm -> FFN
//!    with residual connections throughout.
//!
//! Architecture (Radford et al. 2023, "Robust Speech Recognition via Large-Scale
//! Weak Supervision"):
//! - Pre-norm: LayerNorm before each sub-block
//! - Causal self-attention: standard multi-head attention with causal mask
//! - Cross-attention: Q from decoder, K/V from frozen encoder output
//! - FFN: Linear(D, 4D) -> GELU -> Linear(4D, D)
//! - Residual connections around each sub-block
//!
//! GELU requires CROWN linearization. LayerNorm requires heuristic linearization
//! (IbpValidated mode). Softmax in attention uses piecewise CROWN approximation.
//!
//! Part of #3536: Whisper decoder compose verification tests.

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
// ---------------------------------------------------------------------------

/// Decoder sequence length (number of tokens).
const SEQ_LEN: usize = 4;
/// Embedding / model dimension (tiny Whisper hidden size).
const EMBED_DIM: usize = 64;
/// Number of attention heads (head_dim = EMBED_DIM / NUM_HEADS = 16).
const NUM_HEADS: usize = 4;
/// FFN intermediate dimension: 4x the embedding dimension per Whisper spec.
const FFN_DIM: usize = 256;
/// Encoder output sequence length (e.g., mel frames after conv stems).
const ENC_SEQ_LEN: usize = 8;
/// Small weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Builder helpers: Causal self-attention
// ---------------------------------------------------------------------------

/// Build a causal self-attention kernel (decoder self-attn sub-block).
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable).
/// Output: `[SEQ_LEN, EMBED_DIM]`.
///
/// This is the first sub-block of each Whisper decoder layer:
///   LayerNorm(x) -> MHA(causal) -> + x (residual)
fn build_causal_self_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_dec_causal_self_attn");

    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    let shape = [SEQ_LEN, EMBED_DIM];

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

    b.build(out).expect("valid causal self-attention kernel")
}

/// Bindings for the causal self-attention kernel.
fn causal_self_attention_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // x [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ln_w), // ln_weight [EMBED_DIM]
        TensorParamBinding::ConstantTensor(ln_b), // ln_bias [EMBED_DIM]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj), // out_weight [D, D]
    ]
}

// ---------------------------------------------------------------------------
// Builder helpers: Cross-attention
// ---------------------------------------------------------------------------

/// Build a cross-attention kernel (decoder cross-attn sub-block).
///
/// Q input: `[SEQ_LEN, EMBED_DIM]` (Variable -- decoder hidden state).
/// KV input: `[ENC_SEQ_LEN, EMBED_DIM]` (ConstantTensor -- encoder output).
/// Output: `[SEQ_LEN, EMBED_DIM]`.
///
/// This is the second sub-block of each Whisper decoder layer:
///   LayerNorm(x) -> CrossMHA(x, encoder_output) -> + x (residual)
///
/// The encoder output is bound as Constant because it is certified
/// separately by the encoder composition tests (AC3). The decoder
/// cross-attention treats it as a fixed input with known bounds.
fn build_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_dec_cross_attn");

    let q_input = b.add_input("decoder_hidden", &[SEQ_LEN, EMBED_DIM]);
    let kv_input = b.add_input("encoder_output", &[ENC_SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    let shape = [SEQ_LEN, EMBED_DIM];

    // Pre-norm: LayerNorm on the Q (decoder) side only
    let normed = b.add_layer_norm(q_input, eps, 1, ln_w, ln_b, &shape);

    // Cross-attention: Q from decoder, K/V from encoder
    // Standard (bidirectional) mask -- decoder can attend to all encoder positions
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
        TensorParamBinding::Variable,                 // decoder_hidden [S, D]
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
// Builder helpers: Full decoder block
// ---------------------------------------------------------------------------

/// Build a full Whisper decoder block:
///   SelfAttn(causal) -> LayerNorm -> CrossAttn(encoder) -> LayerNorm -> FFN
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable -- decoder hidden state).
/// Encoder output: `[ENC_SEQ_LEN, EMBED_DIM]` (Constant).
/// Output: `[SEQ_LEN, EMBED_DIM]`.
///
/// Pre-norm structure with 3 residual connections:
/// 1. LN -> MHA(causal) -> + residual
/// 2. LN -> CrossMHA(encoder_output) -> + residual
/// 3. LN -> Linear(D, 4D) -> GELU -> Linear(4D, D) -> + residual
fn build_full_decoder_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_dec_full_block");

    // Inputs
    let q_input = b.add_input("decoder_hidden", &[SEQ_LEN, EMBED_DIM]);
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

    let shape = [SEQ_LEN, EMBED_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // --- Sub-block 1: Causal self-attention ---
    let sa_normed = b.add_layer_norm(q_input, eps, 1, sa_ln_w, sa_ln_b, &shape);
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
            &shape, // output follows Q shape [SEQ_LEN, EMBED_DIM]
        )
        .expect("valid cross-attention");
    let residual2 = b.add_binary_add(residual1, ca_out, &shape);

    // --- Sub-block 3: FFN ---
    let ffn_normed = b.add_layer_norm(residual2, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
    let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
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
        TensorParamBinding::Variable,                 // decoder_hidden [S, D]
        TensorParamBinding::ConstantTensor(kv_const), // encoder_output [E, D]
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
// Tests: Causal self-attention
// ===========================================================================

/// Causal self-attention TensorKernelDef validates.
#[test]
fn test_whisper_dec_causal_self_attn_def_validates() {
    let def = build_causal_self_attention_kernel();
    def.validate()
        .expect("causal self-attention kernel should validate");
}

/// Causal self-attention translates to NY GraphNetwork.
#[test]
fn test_whisper_dec_causal_self_attn_graph_builds() {
    let def = build_causal_self_attention_kernel();
    let bindings = causal_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("causal self-attention graph should translate");

    // LayerNorm + Q/K/V projections + reshape + transpose + attention +
    // transpose + reshape + output projection + residual = many nodes.
    assert!(
        graph.num_nodes() >= 10,
        "causal self-attention graph should have >= 10 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through causal self-attention.
///
/// The causal mask restricts attention to only attend to previous positions,
/// which should produce tighter bounds than bidirectional attention since
/// fewer terms contribute to each output position.
#[test]
fn test_whisper_dec_causal_self_attn_ibp_propagates() {
    let def = build_causal_self_attention_kernel();
    let bindings = causal_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through causal self-attention");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder causal self-attn IBP: bounds=[{lo_min}, {hi_max}]");

    // With small weights (0.02) and [-1, 1] input, bounds should stay reasonable.
    // The residual connection adds input back, so bounds are at least as wide
    // as input bounds.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN propagation through causal self-attention.
///
/// LayerNorm requires heuristic CROWN linearization (IbpValidated mode).
/// Softmax and GELU (within attention scoring) linearize via CROWN.
#[test]
fn test_whisper_dec_causal_self_attn_crown_propagation() {
    let def = build_causal_self_attention_kernel();
    let bindings = causal_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder causal self-attn: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Verify and record causal self-attention under status key.
#[test]
fn test_whisper_dec_causal_self_attn_verify_and_record() {
    let def = build_causal_self_attention_kernel();
    let bindings = causal_self_attention_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_decoder_causal_self_attn");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, EMBED_DIM]);

    // LayerNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Causal self-attn with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: Cross-attention
// ===========================================================================

/// Cross-attention TensorKernelDef validates.
#[test]
fn test_whisper_dec_cross_attn_def_validates() {
    let def = build_cross_attention_kernel();
    def.validate()
        .expect("cross-attention kernel should validate");
}

/// Cross-attention translates to NY GraphNetwork.
#[test]
fn test_whisper_dec_cross_attn_graph_builds() {
    let def = build_cross_attention_kernel();
    let bindings = cross_attention_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("cross-attention graph should translate");

    // LayerNorm + Q projection + K/V projections from encoder +
    // reshape + transpose + attention + transpose + reshape +
    // output projection + residual = many nodes.
    assert!(
        graph.num_nodes() >= 10,
        "cross-attention graph should have >= 10 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through cross-attention.
///
/// Key structural difference from self-attention: Q comes from the Variable
/// decoder hidden state, while K/V come from the Constant encoder output.
/// The output shape follows Q: [SEQ_LEN, EMBED_DIM], not [ENC_SEQ_LEN, ...].
#[test]
fn test_whisper_dec_cross_attn_ibp_propagates() {
    let def = build_cross_attention_kernel();
    let bindings = cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attention");

    // Output shape matches Q (decoder) sequence length, not KV (encoder).
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape must be [SEQ_LEN, EMBED_DIM], not [ENC_SEQ_LEN, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder cross-attn IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through cross-attention.
///
/// Cross-attention with constant K/V should allow CROWN to produce tighter
/// bounds than IBP since the K/V branch has zero perturbation radius.
#[test]
fn test_whisper_dec_cross_attn_crown_propagation() {
    let def = build_cross_attention_kernel();
    let bindings = cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder cross-attn: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// IBP bounds width stays reasonable for cross-attention.
///
/// With small weights (0.02) and [-1, 1] input, bounds should not blow up.
/// Cross-attention with constant encoder output should produce tighter bounds
/// than self-attention because K/V perturbation is zero.
#[test]
fn test_whisper_dec_cross_attn_bounds_width() {
    let def = build_cross_attention_kernel();
    let bindings = cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attention");
    let (lo, hi) = output.lower_upper();

    let max_width = lo
        .iter()
        .zip(hi.iter())
        .map(|(l, u)| (u - l).abs())
        .fold(0.0f32, f32::max);

    // Small weights and bounded input should keep bounds manageable.
    assert!(
        max_width < 500.0,
        "cross-attention IBP bounds max width {max_width} should be < 500.0"
    );
}

/// Verify and record cross-attention under status key.
#[test]
fn test_whisper_dec_cross_attn_verify_and_record() {
    let def = build_cross_attention_kernel();
    let bindings = cross_attention_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_decoder_cross_attn");
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (decoder_hidden)"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, EMBED_DIM]);

    // LayerNorm on Q branch uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Cross-attn with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: Full decoder block
// ===========================================================================

/// Full decoder block TensorKernelDef validates.
#[test]
fn test_whisper_dec_full_block_def_validates() {
    let def = build_full_decoder_block_kernel();
    def.validate()
        .expect("full decoder block kernel should validate");
}

/// Full decoder block translates to NY GraphNetwork.
#[test]
fn test_whisper_dec_full_block_graph_builds() {
    let def = build_full_decoder_block_kernel();
    let bindings = full_decoder_block_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("full decoder block graph should translate");

    // Self-attn: LN + MHA + residual (~12 nodes)
    // Cross-attn: LN + CrossMHA + residual (~12 nodes)
    // FFN: LN + Linear + GELU + Linear + residual (~6 nodes)
    // Total: at least 25 nodes
    assert!(
        graph.num_nodes() >= 25,
        "full decoder block should have >= 25 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full decoder block.
#[test]
fn test_whisper_dec_full_block_ibp_propagates() {
    let def = build_full_decoder_block_kernel();
    let bindings = full_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full decoder block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "decoder block output shape must be [SEQ_LEN, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder full block IBP: bounds=[{lo_min}, {hi_max}]");

    // Bounds may be wide due to 3 chained LayerNorms with residuals.
    // Check finiteness as the primary invariant.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through the full decoder block.
///
/// The full decoder block contains 3 LayerNorms, so CROWN uses heuristic
/// linearization (IbpValidated mode). CROWN may fall back to IBP due to
/// the depth of normalization layers.
#[test]
fn test_whisper_dec_full_block_crown_propagation() {
    let def = build_full_decoder_block_kernel();
    let bindings = full_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder full block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Narrower input produces tighter output bounds (monotonicity).
///
/// IBP monotonicity: narrower input perturbation should produce narrower
/// output bounds for the majority of elements. LayerNorm decomposition may
/// cause some elements to violate strict monotonicity.
#[test]
fn test_whisper_dec_full_block_narrow_inputs_tighter() {
    let def = build_full_decoder_block_kernel();
    let bindings = full_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 10.0);
    let narrow_input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let wide_output = graph.propagate_ibp(&wide_input).expect("wide IBP");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("narrow IBP");

    let (wide_lo, wide_hi) = wide_output.lower_upper();
    let (narrow_lo, narrow_hi) = narrow_output.lower_upper();

    let wide_range = wide_hi.iter().zip(wide_lo.iter()).map(|(h, l)| h - l);
    let narrow_range = narrow_hi.iter().zip(narrow_lo.iter()).map(|(h, l)| h - l);

    // At least half of output elements should have narrower bounds with
    // narrower input (IBP monotonicity may not hold element-wise due to
    // decomposed norm approximations).
    let tighter_count = wide_range.zip(narrow_range).filter(|(w, n)| n <= w).count();
    let total = wide_lo.len();
    assert!(
        tighter_count > total / 2,
        "narrow input should produce tighter bounds for > 50% of elements, got {tighter_count}/{total}"
    );
}

/// Verify and record full decoder block under status key.
#[test]
fn test_whisper_dec_full_block_verify_and_record() {
    let def = build_full_decoder_block_kernel();
    let bindings = full_decoder_block_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_decoder_full_block");
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (decoder_hidden)"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, EMBED_DIM]);

    // 3 LayerNorms use heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Full decoder block with LayerNorms should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
