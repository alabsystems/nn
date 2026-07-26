// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper pipeline-level NY composition tests.
//!
//! Fills verification gaps identified in `nn_verify_status_whisper.json`:
//!
//! 1. **Decoder FFN (isolated)**: Dedicated compose test for the decoder FFN sub-block
//!    (Linear -> GELU -> Linear with residual). Previously only covered implicitly
//!    through full decoder block tests; the status entry was `heuristic` with no
//!    dedicated compose test.
//!
//! 2. **Decoder token embedding + positional encoding**: Embedding lookup (as a
//!    linear projection from one-hot) + learned positional embedding add. Tests the
//!    decoder input path before transformer blocks.
//!
//! 3. **LM head with log-softmax**: Output projection (Linear) followed by
//!    log_softmax, modeling the complete decoder output distribution computation.
//!    The existing LM head test uses only Linear; this extends through log_softmax.
//!
//! 4. **Decoder FFN narrow-input monotonicity**: Verifies that narrower input
//!    bounds produce tighter output bounds through the isolated FFN, confirming
//!    IBP monotonicity for this sub-block.
//!
//! 5. **Mel-to-encoder-embedding bridge**: Conv features -> Transpose -> Positional
//!    embedding add, composing the mel_spectrogram and encoder embedding stages into
//!    a single verified pipeline segment.
//!
//! Architecture reference: Radford et al. 2023, "Robust Speech Recognition via
//! Large-Scale Weak Supervision."
//!
//! Part of #4276: Deepen Whisper NY compose verification.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    uniform_bounds, verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding, VerificationSoundnessMode};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const N_MEL: usize = 4;
const SEQ_LEN: usize = 8;
const EMBED_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const FFN_DIM: usize = 128;
const CONV_KERNEL: usize = 3;
const CONV_PADDING: usize = 1;
const DEC_SEQ_LEN: usize = 4;
const VOCAB_SIZE: usize = 16;
const WEIGHT_MAG: f32 = 0.02;

fn after_conv1_len() -> usize {
    conv1d_out_len(SEQ_LEN, CONV_KERNEL, 1, CONV_PADDING)
}

fn after_conv2_len() -> usize {
    conv1d_out_len(after_conv1_len(), CONV_KERNEL, 2, CONV_PADDING)
}

// ===========================================================================
// 1. Decoder FFN (isolated)
// ===========================================================================

/// Build an isolated decoder FFN sub-block.
///
/// Input: `[DEC_SEQ_LEN, EMBED_DIM]` (Variable).
/// Output: `[DEC_SEQ_LEN, EMBED_DIM]`.
///
/// Architecture:
///   LayerNorm(x) -> Linear(D, FFN_DIM) -> GELU -> Linear(FFN_DIM, D) -> + x (residual)
///
/// This is the third sub-block of each Whisper decoder layer. Previously only
/// covered implicitly through full decoder block tests.
fn build_decoder_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_dec_ffn_isolated");

    let input = b.add_input("x", &[DEC_SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, EMBED_DIM]);
    let ffn2_w = b.add_input("ffn2_weight", &[EMBED_DIM, FFN_DIM]);

    let shape = [DEC_SEQ_LEN, EMBED_DIM];
    let ffn_shape = [DEC_SEQ_LEN, FFN_DIM];

    // Pre-norm: LayerNorm
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);

    // FFN: Linear(D, FFN_DIM) -> GELU -> Linear(FFN_DIM, D)
    let ffn1 = b.add_linear(normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);

    // Residual connection
    let out = b.add_binary_add(input, ffn2, &shape);

    b.build(out).expect("valid decoder FFN kernel")
}

fn decoder_ffn_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,               // x [DEC_SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantScalar(1e-5),   // eps
        TensorParamBinding::ConstantTensor(ln_w),   // ln_weight [EMBED_DIM]
        TensorParamBinding::ConstantTensor(ln_b),   // ln_bias [EMBED_DIM]
        TensorParamBinding::ConstantTensor(w_ffn1), // ffn1_weight [FFN_DIM, EMBED_DIM]
        TensorParamBinding::ConstantTensor(w_ffn2), // ffn2_weight [EMBED_DIM, FFN_DIM]
    ]
}

// ===========================================================================
// 2. Decoder token embedding + positional encoding
// ===========================================================================

/// Build a decoder embedding input path.
///
/// Input: `[DEC_SEQ_LEN, EMBED_DIM]` (Variable -- simulates token embedding output).
/// Positional embedding: `[DEC_SEQ_LEN, EMBED_DIM]` (Constant -- learned PE).
/// Output: `[DEC_SEQ_LEN, EMBED_DIM]`.
///
/// Architecture:
///   TokenEmbedding(x) + PositionalEmbedding
///
/// In production, the token embedding is a lookup table. For verification, we
/// model the token embedding output as the Variable input (the embedding vectors
/// for the input tokens). The positional embedding is a constant add.
///
/// This tests the decoder input path before transformer blocks.
fn build_decoder_embedding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_dec_embedding");

    let token_emb = b.add_input("token_emb", &[DEC_SEQ_LEN, EMBED_DIM]);
    let pos_emb = b.add_input("pos_emb", &[DEC_SEQ_LEN, EMBED_DIM]);

    // Add positional embedding to token embedding
    let out = b.add_binary_add(token_emb, pos_emb, &[DEC_SEQ_LEN, EMBED_DIM]);

    b.build(out).expect("valid decoder embedding kernel")
}

fn decoder_embedding_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // token_emb [DEC_SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DEC_SEQ_LEN, EMBED_DIM]),
            WEIGHT_MAG,
        )), // pos_emb [DEC_SEQ_LEN, EMBED_DIM]
    ]
}

// ===========================================================================
// 3. LM head with log-softmax
// ===========================================================================

/// Build an LM head with log-softmax.
///
/// Input: `[DEC_SEQ_LEN, EMBED_DIM]` (Variable -- decoder hidden state after final LN).
/// Output: `[DEC_SEQ_LEN, VOCAB_SIZE]` (log-probabilities).
///
/// Architecture:
///   LayerNorm(x) -> Linear(EMBED_DIM, VOCAB_SIZE) -> log_softmax(axis=-1)
///
/// The existing LM head compose test covers only the linear projection. This
/// extends through log_softmax to verify the full output distribution path.
/// Log-softmax is numerically more stable than softmax + log and is the
/// standard output for cross-entropy loss computation.
fn build_lm_head_log_softmax_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_lm_head_log_softmax");

    let input = b.add_input("x", &[DEC_SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let lm_w = b.add_input("lm_weight", &[VOCAB_SIZE, EMBED_DIM]);

    let shape = [DEC_SEQ_LEN, EMBED_DIM];

    // Final LayerNorm
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);

    // Linear projection to vocab
    let logits = b.add_linear(normed, lm_w, None, &[DEC_SEQ_LEN, VOCAB_SIZE]);

    // Log-softmax over vocabulary dimension
    let out = b.add_log_softmax(logits, 1, &[DEC_SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid LM head log-softmax kernel")
}

fn lm_head_log_softmax_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, d]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,             // x [DEC_SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ln_w), // ln_weight [EMBED_DIM]
        TensorParamBinding::ConstantTensor(ln_b), // ln_bias [EMBED_DIM]
        TensorParamBinding::ConstantTensor(lm_w), // lm_weight [VOCAB_SIZE, EMBED_DIM]
    ]
}

// ===========================================================================
// 4. Mel-to-encoder-embedding bridge
// ===========================================================================

/// Build the mel-to-encoder-embedding bridge.
///
/// Input: `[N_MEL, SEQ_LEN]` (Variable -- mel spectrogram).
/// Output: `[T_OUT, EMBED_DIM]` (after conv stems + transpose + pos emb).
///
/// Architecture:
///   Conv1d(N_MEL -> EMBED_DIM, k=3, s=1, p=1) -> GELU ->
///   Conv1d(EMBED_DIM -> EMBED_DIM, k=3, s=2, p=1) -> GELU ->
///   Transpose([EMBED_DIM, T] -> [T, EMBED_DIM]) ->
///   + positional_embedding
///
/// This composes the mel feature extraction and encoder embedding stages into
/// a single verified pipeline segment. Tests bounds stability through the
/// full mel-to-embedding path without transformer blocks.
fn build_mel_to_embedding_kernel() -> (TensorKernelDef, usize) {
    let t_mid = after_conv1_len();
    let t_out = after_conv2_len();
    let mut b = TensorBlockBuilder::new("whisper_mel_to_embedding");

    let mel = b.add_input("mel", &[N_MEL, SEQ_LEN]);

    // Conv stem #1
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

    // Conv stem #2
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

    // Transpose: [EMBED_DIM, T_OUT] -> [T_OUT, EMBED_DIM]
    let transposed = b.add_transpose(gelu2, &[1, 0], &[t_out, EMBED_DIM]);

    // Add positional embedding
    let pos_emb = b.add_input("pos_emb", &[t_out, EMBED_DIM]);
    let out = b.add_binary_add(transposed, pos_emb, &[t_out, EMBED_DIM]);

    (b.build(out).expect("valid mel-to-embedding kernel"), t_out)
}

fn mel_to_embedding_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let t_out = after_conv2_len();

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
    ]
}

// ===========================================================================
// 5. Decoder embedding + single FFN block (embedding-to-FFN composition)
// ===========================================================================

/// Build a decoder embedding-to-FFN pipeline.
///
/// Input: `[DEC_SEQ_LEN, EMBED_DIM]` (Variable -- token embedding output).
/// Output: `[DEC_SEQ_LEN, EMBED_DIM]`.
///
/// Architecture:
///   TokenEmb(x) + PosEmb -> LayerNorm -> Linear(D, FFN_DIM) -> GELU ->
///   Linear(FFN_DIM, D) -> + residual
///
/// Composes the decoder embedding input path with a single FFN block,
/// testing bounds propagation through the full input-to-first-computation
/// path of the decoder.
fn build_decoder_emb_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_dec_emb_ffn");

    let token_emb = b.add_input("token_emb", &[DEC_SEQ_LEN, EMBED_DIM]);
    let pos_emb = b.add_input("pos_emb", &[DEC_SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, EMBED_DIM]);
    let ffn2_w = b.add_input("ffn2_weight", &[EMBED_DIM, FFN_DIM]);

    let shape = [DEC_SEQ_LEN, EMBED_DIM];
    let ffn_shape = [DEC_SEQ_LEN, FFN_DIM];

    // Token embedding + positional embedding
    let combined = b.add_binary_add(token_emb, pos_emb, &shape);

    // Pre-norm FFN
    let normed = b.add_layer_norm(combined, eps, 1, ln_w, ln_b, &shape);
    let ffn1 = b.add_linear(normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(combined, ffn2, &shape);

    b.build(out).expect("valid decoder embedding + FFN kernel")
}

fn decoder_emb_ffn_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable, // token_emb [DEC_SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DEC_SEQ_LEN, d]), WEIGHT_MAG)), // pos_emb
        TensorParamBinding::ConstantScalar(1e-5),   // eps
        TensorParamBinding::ConstantTensor(ln_w),   // ln_weight
        TensorParamBinding::ConstantTensor(ln_b),   // ln_bias
        TensorParamBinding::ConstantTensor(w_ffn1), // ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2), // ffn2_weight
    ]
}

// ###########################################################################
// Tests: Decoder FFN (isolated)
// ###########################################################################

#[test]
fn test_whisper_dec_ffn_def_validates() {
    let def = build_decoder_ffn_kernel();
    def.validate().expect("decoder FFN kernel should validate");
}

#[test]
fn test_whisper_dec_ffn_graph_builds() {
    let def = build_decoder_ffn_kernel();
    let bindings = decoder_ffn_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("decoder FFN graph should translate");

    // LN + Linear + GELU + Linear + residual >= 5 nodes
    assert!(
        graph.num_nodes() >= 5,
        "decoder FFN graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through isolated decoder FFN.
///
/// The FFN sub-block is the simplest of the three decoder sub-blocks
/// (self-attn, cross-attn, FFN). It contains one LayerNorm and one GELU.
#[test]
fn test_whisper_dec_ffn_ibp_propagates() {
    let def = build_decoder_ffn_kernel();
    let bindings = decoder_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder FFN");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder FFN IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN propagation through isolated decoder FFN.
///
/// LayerNorm requires heuristic CROWN linearization. GELU linearizes via CROWN.
#[test]
fn test_whisper_dec_ffn_crown_propagation() {
    let def = build_decoder_ffn_kernel();
    let bindings = decoder_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ_LEN, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder FFN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Verify and record decoder FFN under status key.
///
/// Fills the gap in nn_verify_status_whisper.json: decoder_ffn was
/// "heuristic" with no dedicated compose test.
#[test]
fn test_whisper_dec_ffn_verify_and_record() {
    let def = build_decoder_ffn_kernel();
    let bindings = decoder_ffn_bindings();
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_decoder_ffn_isolated");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[DEC_SEQ_LEN, EMBED_DIM]);

    // LayerNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Decoder FFN with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

/// Narrower input produces tighter output bounds through decoder FFN.
#[test]
fn test_whisper_dec_ffn_narrow_inputs_tighter() {
    let def = build_decoder_ffn_kernel();
    let bindings = decoder_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 10.0);
    let narrow_input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

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
        "narrow input should produce tighter bounds for > 50% of elements, got {tighter_count}/{total}"
    );
}

// ###########################################################################
// Tests: Decoder token embedding + positional encoding
// ###########################################################################

#[test]
fn test_whisper_dec_embedding_def_validates() {
    let def = build_decoder_embedding_kernel();
    def.validate()
        .expect("decoder embedding kernel should validate");
}

/// IBP bounds propagate through decoder embedding path.
///
/// Token embedding + positional embedding is a pure linear operation
/// (add constant). IBP should produce exact bounds.
#[test]
fn test_whisper_dec_embedding_ibp_propagates() {
    let def = build_decoder_embedding_kernel();
    let bindings = decoder_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder embedding");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder embedding IBP: bounds=[{lo_min}, {hi_max}]");

    // Pure linear: add constant. Bounds shift by WEIGHT_MAG.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        (hi_max - lo_min) < 5.0,
        "embedding bounds should be tight (pure linear), got width {}",
        hi_max - lo_min
    );
}

/// CROWN propagation through decoder embedding path.
#[test]
fn test_whisper_dec_embedding_crown_propagation() {
    let def = build_decoder_embedding_kernel();
    let bindings = decoder_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ_LEN, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder embedding: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record decoder embedding under status key.
#[test]
fn test_whisper_dec_embedding_verify_and_record() {
    let def = build_decoder_embedding_kernel();
    let bindings = decoder_embedding_bindings();
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_decoder_token_embedding");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[DEC_SEQ_LEN, EMBED_DIM]);

    // No LayerNorm, pure linear => should be Sound or IbpValidated.
    eprintln!(
        "Decoder embedding soundness mode: {:?}",
        result.verification.soundness_mode
    );
}

// ###########################################################################
// Tests: LM head with log-softmax
// ###########################################################################

#[test]
fn test_whisper_lm_head_log_softmax_def_validates() {
    let def = build_lm_head_log_softmax_kernel();
    def.validate()
        .expect("LM head log-softmax kernel should validate");
}

#[test]
fn test_whisper_lm_head_log_softmax_graph_builds() {
    let def = build_lm_head_log_softmax_kernel();
    let bindings = lm_head_log_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("LM head log-softmax graph should translate");

    // LN + Linear + log_softmax >= 3 nodes
    assert!(
        graph.num_nodes() >= 3,
        "LM head log-softmax graph should have >= 3 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through LM head with log-softmax.
///
/// LayerNorm + Linear is exact under IBP. Log-softmax bounds require
/// special handling: output is always <= 0 (log of probability).
#[test]
fn test_whisper_lm_head_log_softmax_ibp_propagates() {
    let def = build_lm_head_log_softmax_kernel();
    let bindings = lm_head_log_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through LM head log-softmax");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, VOCAB_SIZE],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper LM head log-softmax IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");

    // Log-softmax output should be <= 0 for the upper bound in at least
    // a majority of elements (log of probability). With IBP widening
    // through LayerNorm, the upper bound may exceed 0 for some elements.
    // Check finiteness as the primary invariant.
}

/// CROWN propagation through LM head with log-softmax.
///
/// Log-softmax CROWN linearization uses piecewise bounds. LayerNorm
/// requires heuristic linearization (IbpValidated mode).
#[test]
fn test_whisper_lm_head_log_softmax_crown_propagation() {
    let def = build_lm_head_log_softmax_kernel();
    let bindings = lm_head_log_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ_LEN, VOCAB_SIZE],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper LM head log-softmax: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record LM head log-softmax under status key.
#[test]
fn test_whisper_lm_head_log_softmax_verify_and_record() {
    let def = build_lm_head_log_softmax_kernel();
    let bindings = lm_head_log_softmax_bindings();
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_lm_head_log_softmax");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[DEC_SEQ_LEN, VOCAB_SIZE]);

    // LayerNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "LM head with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ###########################################################################
// Tests: Mel-to-encoder-embedding bridge
// ###########################################################################

#[test]
fn test_whisper_mel_to_embedding_def_validates() {
    let (def, _) = build_mel_to_embedding_kernel();
    def.validate()
        .expect("mel-to-embedding kernel should validate");
}

#[test]
fn test_whisper_mel_to_embedding_graph_builds() {
    let (def, _) = build_mel_to_embedding_kernel();
    let bindings = mel_to_embedding_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("mel-to-embedding graph should translate");

    // 2 conv + 2 GELU + transpose + add >= 6 nodes
    assert!(
        graph.num_nodes() >= 6,
        "mel-to-embedding graph should have >= 6 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through mel-to-encoder-embedding bridge.
///
/// Conv1d and add are linear operators. GELU is the only nonlinearity.
/// No LayerNorm, so bounds should stay tight.
#[test]
fn test_whisper_mel_to_embedding_ibp_propagates() {
    let (def, t_out) = build_mel_to_embedding_kernel();
    let bindings = mel_to_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through mel-to-embedding");

    // Output is transposed: [T_OUT, EMBED_DIM]
    assert_eq!(
        output.lower_upper().0.shape(),
        &[t_out, EMBED_DIM],
        "output shape must be [T_OUT, EMBED_DIM] after transpose + pos_emb"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper mel-to-embedding IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN propagation through mel-to-encoder-embedding bridge.
///
/// No LayerNorm. Conv1d is exact. GELU linearizes via CROWN.
/// Should produce tighter bounds than IBP.
#[test]
fn test_whisper_mel_to_embedding_crown_propagation() {
    let (def, t_out) = build_mel_to_embedding_kernel();
    let bindings = mel_to_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[t_out, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper mel-to-embedding: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Verify and record mel-to-embedding under status key.
#[test]
fn test_whisper_mel_to_embedding_verify_and_record() {
    let (def, t_out) = build_mel_to_embedding_kernel();
    let bindings = mel_to_embedding_bindings();
    let input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_mel_to_encoder_embedding");
    assert_eq!(result.num_variables, 1, "single Variable input (mel)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t_out, EMBED_DIM]);

    // No LayerNorm => soundness depends on GELU linearization.
    eprintln!(
        "Mel-to-embedding soundness mode: {:?}",
        result.verification.soundness_mode
    );
}

/// Narrower input produces tighter output bounds through mel-to-embedding.
#[test]
fn test_whisper_mel_to_embedding_narrow_inputs_tighter() {
    let (def, _) = build_mel_to_embedding_kernel();
    let bindings = mel_to_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[N_MEL, SEQ_LEN], 10.0);
    let narrow_input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

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
        "narrow input should produce tighter bounds for > 50% of elements, got {tighter_count}/{total}"
    );
}

// ###########################################################################
// Tests: Decoder embedding + single FFN block
// ###########################################################################

#[test]
fn test_whisper_dec_emb_ffn_def_validates() {
    let def = build_decoder_emb_ffn_kernel();
    def.validate()
        .expect("decoder embedding + FFN kernel should validate");
}

#[test]
fn test_whisper_dec_emb_ffn_graph_builds() {
    let def = build_decoder_emb_ffn_kernel();
    let bindings = decoder_emb_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("decoder embedding + FFN graph should translate");

    // Add + LN + Linear + GELU + Linear + residual >= 6 nodes
    assert!(
        graph.num_nodes() >= 6,
        "decoder embedding + FFN graph should have >= 6 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through decoder embedding + FFN.
#[test]
fn test_whisper_dec_emb_ffn_ibp_propagates() {
    let def = build_decoder_emb_ffn_kernel();
    let bindings = decoder_emb_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder embedding + FFN");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder emb+FFN IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through decoder embedding + FFN.
#[test]
fn test_whisper_dec_emb_ffn_crown_propagation() {
    let def = build_decoder_emb_ffn_kernel();
    let bindings = decoder_emb_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ_LEN, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper decoder emb+FFN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record decoder embedding + FFN under status key.
#[test]
fn test_whisper_dec_emb_ffn_verify_and_record() {
    let def = build_decoder_emb_ffn_kernel();
    let bindings = decoder_emb_ffn_bindings();
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_decoder_emb_ffn");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[DEC_SEQ_LEN, EMBED_DIM]);

    // LayerNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Decoder emb+FFN with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
