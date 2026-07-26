// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Whisper encoder NY composition.
//!
//! Verifies bounds propagation through the four key Whisper encoder sub-blocks:
//!
//! 1. **Conv1d feature extraction**: Two 1D convolutions (N_MEL->EMBED_DIM,
//!    kernel=3, stride=1, padding=1; then EMBED_DIM->EMBED_DIM, kernel=3,
//!    stride=2, padding=1) with GELU activations.
//!
//! 2. **Sinusoidal positional encoding**: Transpose from [EMBED_DIM, T] to
//!    [T, EMBED_DIM], then add learned/fixed positional embeddings.
//!
//! 3. **Encoder self-attention block**: Pre-norm multi-head self-attention
//!    (Standard mask, not Causal) + LayerNorm + FFN with residual connections.
//!
//! 4. **Full single-encoder-layer pipeline**: Conv features + pos encoding +
//!    1 transformer block + final LayerNorm.
//!
//! Architecture (Radford et al. 2023, "Robust Speech Recognition via Large-Scale
//! Weak Supervision"):
//! - Conv stem: 2x Conv1d with GELU (feature extraction from mel spectrogram)
//! - Pre-norm transformer: LayerNorm before each sub-block
//! - Standard (bidirectional) self-attention (encoder sees all positions)
//! - FFN: Linear(D, 4D) -> GELU -> Linear(4D, D)
//! - Residual connections around each sub-block
//!
//! GELU requires CROWN linearization. LayerNorm requires heuristic linearization
//! (IbpValidated mode). Conv1d is a linear operator for IBP/CROWN.
//!
//! Part of #3558: Whisper encoder compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    uniform_bounds, verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding, VerificationSoundnessMode};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Number of mel frequency bins (production: 128).
const N_MEL: usize = 4;
/// Encoder sequence length of mel input frames (production: 3000).
const SEQ_LEN: usize = 8;
/// Embedding / model dimension (tiny Whisper hidden size).
const EMBED_DIM: usize = 32;
/// Number of attention heads (head_dim = EMBED_DIM / NUM_HEADS = 8).
const NUM_HEADS: usize = 4;
/// FFN intermediate dimension: 4x the embedding dimension per Whisper spec.
const FFN_DIM: usize = 128;
/// Conv1d kernel size for both stems.
const CONV_KERNEL: usize = 3;
/// Conv1d padding for both stems.
const CONV_PADDING: usize = 1;
/// Small weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

/// Output sequence length after the first conv (stride=1, same padding).
fn after_conv1_len() -> usize {
    conv1d_out_len(SEQ_LEN, CONV_KERNEL, 1, CONV_PADDING)
}

/// Output sequence length after the second conv (stride=2, same padding).
fn after_conv2_len() -> usize {
    conv1d_out_len(after_conv1_len(), CONV_KERNEL, 2, CONV_PADDING)
}

// ---------------------------------------------------------------------------
// Builder helpers: Conv1d feature extraction
// ---------------------------------------------------------------------------

/// Build the Conv1d feature extraction sub-block.
///
/// Input: `[N_MEL, SEQ_LEN]` (Variable -- mel spectrogram).
/// Output: `[EMBED_DIM, T_OUT]` where T_OUT = after_conv2_len().
///
/// Architecture:
///   Conv1d(N_MEL -> EMBED_DIM, k=3, s=1, p=1) -> GELU ->
///   Conv1d(EMBED_DIM -> EMBED_DIM, k=3, s=2, p=1) -> GELU
fn build_conv_feature_extraction_kernel() -> (TensorKernelDef, usize) {
    let t_mid = after_conv1_len();
    let t_out = after_conv2_len();
    let mut b = TensorBlockBuilder::new("whisper_enc_conv_features");

    let mel = b.add_input("mel", &[N_MEL, SEQ_LEN]);

    // Conv stem #1: Conv1d(N_MEL -> EMBED_DIM, k=3, s=1, p=1) -> GELU
    let conv1_w = b.add_input("conv1_weight", &[EMBED_DIM, N_MEL, CONV_KERNEL]);
    let conv1_b = b.add_input("conv1_bias", &[EMBED_DIM]);
    let conv1 = b.add_conv1d(
        mel,
        conv1_w,
        Some(conv1_b),
        1,
        CONV_PADDING,
        &[EMBED_DIM, t_mid],
    );
    let gelu1 = b.add_gelu(conv1, &[EMBED_DIM, t_mid]);

    // Conv stem #2: Conv1d(EMBED_DIM -> EMBED_DIM, k=3, s=2, p=1) -> GELU
    let conv2_w = b.add_input("conv2_weight", &[EMBED_DIM, EMBED_DIM, CONV_KERNEL]);
    let conv2_b = b.add_input("conv2_bias", &[EMBED_DIM]);
    let conv2 = b.add_conv1d(
        gelu1,
        conv2_w,
        Some(conv2_b),
        2,
        CONV_PADDING,
        &[EMBED_DIM, t_out],
    );
    let out = b.add_gelu(conv2, &[EMBED_DIM, t_out]);

    (
        b.build(out).expect("valid conv feature extraction kernel"),
        t_out,
    )
}

/// Bindings for the conv feature extraction kernel.
fn conv_feature_extraction_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // mel [N_MEL, SEQ_LEN]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, N_MEL, CONV_KERNEL]),
            WEIGHT_MAG,
        )), // conv1_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)), // conv1_bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, EMBED_DIM, CONV_KERNEL]),
            WEIGHT_MAG,
        )), // conv2_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)), // conv2_bias
    ]
}

// ---------------------------------------------------------------------------
// Builder helpers: Positional encoding
// ---------------------------------------------------------------------------

/// Build the positional encoding sub-block.
///
/// Input: `[EMBED_DIM, T_OUT]` (Variable -- conv features before transpose).
/// Output: `[T_OUT, EMBED_DIM]` (after transpose + positional embedding add).
///
/// Architecture:
///   Transpose([EMBED_DIM, T_OUT] -> [T_OUT, EMBED_DIM]) ->
///   Add(positional_embedding)
fn build_positional_encoding_kernel() -> (TensorKernelDef, usize) {
    let t_out = after_conv2_len();
    let mut b = TensorBlockBuilder::new("whisper_enc_pos_encoding");

    let features = b.add_input("features", &[EMBED_DIM, t_out]);
    let pos_emb = b.add_input("pos_emb", &[t_out, EMBED_DIM]);

    // Transpose: [EMBED_DIM, T_OUT] -> [T_OUT, EMBED_DIM]
    let transposed = b.add_transpose(features, &[1, 0], &[t_out, EMBED_DIM]);

    // Add positional embedding
    let out = b.add_binary_add(transposed, pos_emb, &[t_out, EMBED_DIM]);

    (
        b.build(out).expect("valid positional encoding kernel"),
        t_out,
    )
}

/// Bindings for the positional encoding kernel.
fn positional_encoding_bindings() -> Vec<TensorParamBinding> {
    let t_out = after_conv2_len();
    vec![
        TensorParamBinding::Variable, // features [EMBED_DIM, T_OUT]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[t_out, EMBED_DIM]),
            WEIGHT_MAG,
        )), // pos_emb [T_OUT, EMBED_DIM]
    ]
}

// ---------------------------------------------------------------------------
// Builder helpers: Encoder self-attention block
// ---------------------------------------------------------------------------

/// Build an encoder self-attention block.
///
/// Input: `[T_OUT, EMBED_DIM]` (Variable).
/// Output: `[T_OUT, EMBED_DIM]`.
///
/// This is a single pre-norm encoder transformer block:
///   LayerNorm(x) -> MHA(standard/bidirectional) -> + x (residual)
///   LayerNorm(y) -> Linear(D, 4D) -> GELU -> Linear(4D, D) -> + y (residual)
///
/// Unlike the decoder, encoder self-attention uses Standard (not Causal) mask
/// because the encoder can attend to all positions bidirectionally.
fn build_encoder_self_attention_kernel() -> (TensorKernelDef, usize) {
    let t_out = after_conv2_len();
    let mut b = TensorBlockBuilder::new("whisper_enc_self_attn_block");

    let input = b.add_input("x", &[t_out, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    // Self-attention sub-block weights
    let sa_ln_w = b.add_input("sa_ln_weight", &[EMBED_DIM]);
    let sa_ln_b = b.add_input("sa_ln_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    // FFN sub-block weights
    let ffn_ln_w = b.add_input("ffn_ln_weight", &[EMBED_DIM]);
    let ffn_ln_b = b.add_input("ffn_ln_bias", &[EMBED_DIM]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, EMBED_DIM]);
    let ffn2_w = b.add_input("ffn2_weight", &[EMBED_DIM, FFN_DIM]);

    let shape = [t_out, EMBED_DIM];
    let ffn_shape = [t_out, FFN_DIM];

    // --- Sub-block 1: Pre-norm self-attention ---
    let sa_normed = b.add_layer_norm(input, eps, 1, sa_ln_w, sa_ln_b, &shape);
    let sa_out = b
        .add_multi_head_attention(
            sa_normed,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard, // bidirectional, not causal
            &shape,
        )
        .expect("valid encoder self-attention");
    let residual1 = b.add_binary_add(input, sa_out, &shape);

    // --- Sub-block 2: Pre-norm FFN ---
    let ffn_normed = b.add_layer_norm(residual1, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
    let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(residual1, ffn2, &shape);

    (
        b.build(out)
            .expect("valid encoder self-attention block kernel"),
        t_out,
    )
}

/// Bindings for the encoder self-attention block kernel.
fn encoder_self_attention_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                     // x [T_OUT, EMBED_DIM]
        TensorParamBinding::ConstantScalar(1e-5),         // eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // sa_ln_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // sa_ln_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight
        TensorParamBinding::ConstantTensor(w_proj),       // out_weight
        TensorParamBinding::ConstantTensor(ln_w),         // ffn_ln_weight
        TensorParamBinding::ConstantTensor(ln_b),         // ffn_ln_bias
        TensorParamBinding::ConstantTensor(w_ffn1),       // ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2),       // ffn2_weight
    ]
}

// ---------------------------------------------------------------------------
// Builder helpers: Full single-encoder-layer pipeline
// ---------------------------------------------------------------------------

/// Build a full single-encoder-layer pipeline.
///
/// Input: `[N_MEL, SEQ_LEN]` (Variable -- mel spectrogram).
/// Output: `[T_OUT, EMBED_DIM]`.
///
/// Architecture:
///   Conv1d(N_MEL->D, k=3, s=1, p=1) -> GELU ->
///   Conv1d(D->D, k=3, s=2, p=1) -> GELU ->
///   Transpose([D, T] -> [T, D]) ->
///   + positional_embedding ->
///     1 x Transformer Block (LN + MHA(standard) + residual + LN + FFN + residual) ->
///     Final LayerNorm
fn build_full_encoder_layer_kernel() -> (TensorKernelDef, usize) {
    let t_mid = after_conv1_len();
    let t_out = after_conv2_len();
    let mut b = TensorBlockBuilder::new("whisper_enc_full_layer");

    // --- Variable input: mel spectrogram ---
    let mel = b.add_input("mel", &[N_MEL, SEQ_LEN]);

    // --- Conv stem #1 ---
    let conv1_w = b.add_input("conv1_weight", &[EMBED_DIM, N_MEL, CONV_KERNEL]);
    let conv1_b = b.add_input("conv1_bias", &[EMBED_DIM]);
    let conv1 = b.add_conv1d(
        mel,
        conv1_w,
        Some(conv1_b),
        1,
        CONV_PADDING,
        &[EMBED_DIM, t_mid],
    );
    let gelu1 = b.add_gelu(conv1, &[EMBED_DIM, t_mid]);

    // --- Conv stem #2 ---
    let conv2_w = b.add_input("conv2_weight", &[EMBED_DIM, EMBED_DIM, CONV_KERNEL]);
    let conv2_b = b.add_input("conv2_bias", &[EMBED_DIM]);
    let conv2 = b.add_conv1d(
        gelu1,
        conv2_w,
        Some(conv2_b),
        2,
        CONV_PADDING,
        &[EMBED_DIM, t_out],
    );
    let gelu2 = b.add_gelu(conv2, &[EMBED_DIM, t_out]);

    // --- Transpose + positional embedding ---
    let transposed = b.add_transpose(gelu2, &[1, 0], &[t_out, EMBED_DIM]);
    let pos_emb = b.add_input("pos_emb", &[t_out, EMBED_DIM]);
    let x = b.add_binary_add(transposed, pos_emb, &[t_out, EMBED_DIM]);

    let shape = [t_out, EMBED_DIM];
    let ffn_shape = [t_out, FFN_DIM];

    // --- Single transformer block ---
    let eps = b.add_input("eps", &[1]);
    let sa_ln_w = b.add_input("sa_ln_weight", &[EMBED_DIM]);
    let sa_ln_b = b.add_input("sa_ln_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);
    let ffn_ln_w = b.add_input("ffn_ln_weight", &[EMBED_DIM]);
    let ffn_ln_b = b.add_input("ffn_ln_bias", &[EMBED_DIM]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, EMBED_DIM]);
    let ffn2_w = b.add_input("ffn2_weight", &[EMBED_DIM, FFN_DIM]);

    // Pre-norm self-attention
    let sa_normed = b.add_layer_norm(x, eps, 1, sa_ln_w, sa_ln_b, &shape);
    let sa_out = b
        .add_multi_head_attention(
            sa_normed,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid encoder self-attention");
    let residual1 = b.add_binary_add(x, sa_out, &shape);

    // Pre-norm FFN
    let ffn_normed = b.add_layer_norm(residual1, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
    let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let residual2 = b.add_binary_add(residual1, ffn2, &shape);

    // --- Final LayerNorm ---
    let final_ln_w = b.add_input("final_ln_weight", &[EMBED_DIM]);
    let final_ln_b = b.add_input("final_ln_bias", &[EMBED_DIM]);
    let final_eps = b.add_input("final_eps", &[1]);
    let output = b.add_layer_norm(residual2, final_eps, 1, final_ln_w, final_ln_b, &shape);

    (
        b.build(output).expect("valid full encoder layer kernel"),
        t_out,
    )
}

/// Bindings for the full single-encoder-layer pipeline.
fn full_encoder_layer_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let t_out = after_conv2_len();
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable, // mel [N_MEL, SEQ_LEN]
        // Conv stem #1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d, N_MEL, CONV_KERNEL]),
            WEIGHT_MAG,
        )), // conv1_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)), // conv1_bias
        // Conv stem #2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d, d, CONV_KERNEL]),
            WEIGHT_MAG,
        )), // conv2_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)), // conv2_bias
        // Positional embedding
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[t_out, d]), WEIGHT_MAG)), // pos_emb
        // Transformer block
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // sa_ln_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // sa_ln_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight
        TensorParamBinding::ConstantTensor(w_proj), // out_weight
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ffn_ln_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ffn_ln_bias
        TensorParamBinding::ConstantTensor(w_ffn1), // ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2), // ffn2_weight
        // Final LayerNorm
        TensorParamBinding::ConstantTensor(ln_w), // final_ln_weight
        TensorParamBinding::ConstantTensor(ln_b), // final_ln_bias
        TensorParamBinding::ConstantScalar(1e-5), // final_eps
    ]
}

// ===========================================================================
// Tests: Conv1d feature extraction
// ===========================================================================

/// Conv1d feature extraction TensorKernelDef validates.
#[test]
fn test_whisper_enc_conv_features_def_validates() {
    let (def, _) = build_conv_feature_extraction_kernel();
    def.validate()
        .expect("conv feature extraction kernel should validate");
}

/// Conv1d feature extraction translates to NY GraphNetwork.
#[test]
fn test_whisper_enc_conv_features_graph_builds() {
    let (def, t_out) = build_conv_feature_extraction_kernel();
    let bindings = conv_feature_extraction_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("conv features graph should translate");

    // 2 conv layers + 2 GELU activations = at least 4 graph nodes
    assert!(
        graph.num_nodes() >= 4,
        "conv features graph should have >= 4 nodes, got {}",
        graph.num_nodes()
    );
    assert!(t_out > 0, "output sequence length should be > 0");
}

/// IBP bounds propagate through Conv1d feature extraction.
///
/// Conv1d is a linear operator, so IBP produces exact bounds. GELU is the
/// only nonlinearity -- it is monotonically increasing for x > 0 and bounded
/// below by ~-0.17 at x ~ -0.75, so bounds stay tight.
#[test]
fn test_whisper_enc_conv_features_ibp_propagates() {
    let (def, t_out) = build_conv_feature_extraction_kernel();
    let bindings = conv_feature_extraction_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through conv features");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[EMBED_DIM, t_out],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper encoder conv features IBP: bounds=[{lo_min}, {hi_max}]");

    // With small weights (0.02) and [-1, 1] input, bounds should stay tight.
    // No normalization layers => no bound explosion.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN propagation through Conv1d feature extraction.
///
/// Conv1d is linear, so CROWN is exact. GELU linearization via CROWN
/// should produce tighter bounds than IBP.
#[test]
fn test_whisper_enc_conv_features_crown_propagation() {
    let (def, t_out) = build_conv_feature_extraction_kernel();
    let bindings = conv_feature_extraction_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[EMBED_DIM, t_out],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper encoder conv features: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Verify and record conv feature extraction under status key.
#[test]
fn test_whisper_enc_conv_features_verify_and_record() {
    let (def, t_out) = build_conv_feature_extraction_kernel();
    let bindings = conv_feature_extraction_bindings();
    let input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_encoder_conv_features");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[EMBED_DIM, t_out]);

    // No LayerNorm => should be Sound (conv + GELU are CROWN-compatible).
    // If NY reports Heuristic (e.g., due to GELU approximation),
    // that is acceptable but less ideal.
    eprintln!(
        "Conv features soundness mode: {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: Positional encoding
// ===========================================================================

/// Positional encoding TensorKernelDef validates.
#[test]
fn test_whisper_enc_pos_encoding_def_validates() {
    let (def, _) = build_positional_encoding_kernel();
    def.validate()
        .expect("positional encoding kernel should validate");
}

/// Positional encoding translates to NY GraphNetwork.
#[test]
fn test_whisper_enc_pos_encoding_graph_builds() {
    let (def, _) = build_positional_encoding_kernel();
    let bindings = positional_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("positional encoding graph should translate");

    // Transpose + Add = at least 2 graph nodes
    assert!(
        graph.num_nodes() >= 2,
        "positional encoding graph should have >= 2 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through positional encoding.
///
/// Transpose is a shape-only operation (no value change). Adding a constant
/// positional embedding shifts bounds uniformly. Both are exact under IBP.
#[test]
fn test_whisper_enc_pos_encoding_ibp_propagates() {
    let (def, t_out) = build_positional_encoding_kernel();
    let bindings = positional_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[EMBED_DIM, t_out], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through positional encoding");

    // Output is transposed: [T_OUT, EMBED_DIM]
    assert_eq!(
        output.lower_upper().0.shape(),
        &[t_out, EMBED_DIM],
        "output shape must be [T_OUT, EMBED_DIM] after transpose"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper encoder pos encoding IBP: bounds=[{lo_min}, {hi_max}]");

    // Transpose + constant add: bounds shift by WEIGHT_MAG (0.02).
    // Input is [-1, 1], so output should be approximately [-1+W, 1+W].
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        (hi_max - lo_min) < 5.0,
        "pos encoding bounds should be tight (no nonlinearities), got width {}",
        hi_max - lo_min
    );
}

/// CROWN propagation through positional encoding.
///
/// Transpose and constant addition are both linear, so CROWN should produce
/// the same bounds as IBP.
#[test]
fn test_whisper_enc_pos_encoding_crown_propagation() {
    let (def, t_out) = build_positional_encoding_kernel();
    let bindings = positional_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[EMBED_DIM, t_out], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[t_out, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper encoder pos encoding: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record positional encoding under status key.
#[test]
fn test_whisper_enc_pos_encoding_verify_and_record() {
    let (def, t_out) = build_positional_encoding_kernel();
    let bindings = positional_encoding_bindings();
    let input = uniform_bounds(&[EMBED_DIM, t_out], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_encoder_pos_encoding");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t_out, EMBED_DIM]);

    // Pure linear ops (transpose + add) should produce Sound verification.
    // If NY reports Heuristic, that is acceptable.
    eprintln!(
        "Positional encoding soundness mode: {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: Encoder self-attention block
// ===========================================================================

/// Encoder self-attention block TensorKernelDef validates.
#[test]
fn test_whisper_enc_self_attn_def_validates() {
    let (def, _) = build_encoder_self_attention_kernel();
    def.validate()
        .expect("encoder self-attention block kernel should validate");
}

/// Encoder self-attention block translates to NY GraphNetwork.
#[test]
fn test_whisper_enc_self_attn_graph_builds() {
    let (def, _) = build_encoder_self_attention_kernel();
    let bindings = encoder_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("encoder self-attention graph should translate");

    // LN + MHA(Q/K/V proj + reshape + attn + reshape + out proj) + residual +
    // LN + FFN(2 linears + GELU) + residual = at least 15 nodes
    assert!(
        graph.num_nodes() >= 15,
        "encoder self-attention graph should have >= 15 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through encoder self-attention block.
///
/// Unlike decoder self-attention, encoder uses Standard (bidirectional) mask,
/// meaning every position attends to every other position. This may produce
/// slightly wider bounds than causal attention since more terms contribute.
#[test]
fn test_whisper_enc_self_attn_ibp_propagates() {
    let (def, t_out) = build_encoder_self_attention_kernel();
    let bindings = encoder_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[t_out, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder self-attention");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[t_out, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper encoder self-attn IBP: bounds=[{lo_min}, {hi_max}]");

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

/// CROWN propagation through encoder self-attention block.
///
/// LayerNorm requires heuristic CROWN linearization (IbpValidated mode).
/// Softmax and GELU linearize via CROWN.
#[test]
fn test_whisper_enc_self_attn_crown_propagation() {
    let (def, t_out) = build_encoder_self_attention_kernel();
    let bindings = encoder_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[t_out, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[t_out, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper encoder self-attn: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Verify and record encoder self-attention block under status key.
#[test]
fn test_whisper_enc_self_attn_verify_and_record() {
    let (def, t_out) = build_encoder_self_attention_kernel();
    let bindings = encoder_self_attention_bindings();
    let input = uniform_bounds(&[t_out, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_encoder_self_attn_block");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t_out, EMBED_DIM]);

    // LayerNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Encoder self-attn with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: Full single-encoder-layer pipeline
// ===========================================================================

/// Full encoder layer TensorKernelDef validates.
#[test]
fn test_whisper_enc_full_layer_def_validates() {
    let (def, _) = build_full_encoder_layer_kernel();
    def.validate()
        .expect("full encoder layer kernel should validate");
}

/// Full encoder layer translates to NY GraphNetwork.
#[test]
fn test_whisper_enc_full_layer_graph_builds() {
    let (def, _) = build_full_encoder_layer_kernel();
    let bindings = full_encoder_layer_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("full encoder layer graph should translate");

    // Conv stems (2 conv + 2 GELU) + transpose + pos_emb add +
    // transformer block (LN + MHA + residual + LN + FFN + residual) +
    // final LayerNorm = at least 20 nodes
    assert!(
        graph.num_nodes() >= 20,
        "full encoder layer graph should have >= 20 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full encoder layer.
#[test]
fn test_whisper_enc_full_layer_ibp_propagates() {
    let (def, t_out) = build_full_encoder_layer_kernel();
    let bindings = full_encoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full encoder layer");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[t_out, EMBED_DIM],
        "encoder layer output shape must be [T_OUT, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper encoder full layer IBP: bounds=[{lo_min}, {hi_max}]");

    // Bounds may be wide due to LayerNorm + attention composition.
    // Check finiteness as the primary invariant.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through the full encoder layer.
///
/// The full encoder layer contains 3 LayerNorms (2 in transformer block +
/// 1 final), so CROWN uses heuristic linearization (IbpValidated mode).
/// CROWN may fall back to IBP due to the depth of normalization layers.
#[test]
fn test_whisper_enc_full_layer_crown_propagation() {
    let (def, t_out) = build_full_encoder_layer_kernel();
    let bindings = full_encoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[t_out, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper encoder full layer: method={method:?}, bounds=[{lo_min}, {hi_max}]");
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
fn test_whisper_enc_full_layer_narrow_inputs_tighter() {
    let (def, _) = build_full_encoder_layer_kernel();
    let bindings = full_encoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[N_MEL, SEQ_LEN], 10.0);
    let narrow_input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

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

/// Verify and record full encoder layer under status key.
#[test]
fn test_whisper_enc_full_layer_verify_and_record() {
    let (def, t_out) = build_full_encoder_layer_kernel();
    let bindings = full_encoder_layer_bindings();
    let input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_encoder_full_layer");
    assert_eq!(result.num_variables, 1, "single Variable input (mel)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t_out, EMBED_DIM]);

    // 3 LayerNorms use heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Full encoder layer with LayerNorms should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
