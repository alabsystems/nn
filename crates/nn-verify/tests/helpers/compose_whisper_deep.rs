// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep NY composition tests for the Whisper speech-to-text model.
//!
//! Covers 10 architectural sub-blocks with IBP and CROWN verification:
//!
//! 1. **Mel spectrogram processing**: Conv1d -> GELU -> Conv1d feature extraction
//! 2. **Encoder self-attention**: Multi-head attention with sinusoidal PE
//! 3. **Encoder FFN**: Linear -> GELU -> Linear with residual connection
//! 4. **Encoder layer**: Self-attention -> LayerNorm -> FFN -> LayerNorm
//! 5. **Decoder cross-attention**: Decoder queries attend to encoder output
//! 6. **Decoder causal self-attention**: Causal mask preserves attention bounds
//! 7. **Full encoder block**: 2-layer stacked encoder (IBP + CROWN)
//! 8. **Token embedding + positional**: Embedding lookup + learned PE
//! 9. **LM head**: Linear -> softmax for token prediction
//! 10. **End-to-end encoder**: Mel -> Conv -> N encoder layers -> LayerNorm
//!
//! Architecture reference: Radford et al. 2023, "Robust Speech Recognition via
//! Large-Scale Weak Supervision."
//!
//! GELU requires CROWN linearization. LayerNorm requires heuristic linearization
//! (IbpValidated mode per nn_engineering.md). Conv1d and Linear are exact under IBP.
//! Softmax uses piecewise CROWN approximation.
//!
//! Part of #3942: Whisper deep compose verification tests.

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
/// Encoder input sequence length of mel frames (production: 3000).
const MEL_SEQ_LEN: usize = 8;
/// Embedding / model dimension.
const D_MODEL: usize = 16;
/// Number of attention heads (head_dim = D_MODEL / N_HEADS = 4).
const N_HEADS: usize = 4;
/// FFN intermediate dimension: 4x the embedding dimension per Whisper spec.
const FFN_DIM: usize = 64;
/// Conv1d kernel size for encoder stems.
const CONV_KERNEL: usize = 3;
/// Conv1d padding for encoder stems.
const CONV_PADDING: usize = 1;
/// Decoder sequence length (number of tokens).
const DEC_SEQ: usize = 4;
/// Encoder output sequence length (mel frames after conv stems).
const ENC_SEQ: usize = 6;
/// Vocabulary size for LM head.
const VOCAB_SIZE: usize = 32;
/// Small weight magnitude for bounded verification.
const W_MAG: f32 = 0.02;

/// Output sequence length after the first conv (stride=1, same padding).
fn after_conv1() -> usize {
    conv1d_out_len(MEL_SEQ_LEN, CONV_KERNEL, 1, CONV_PADDING)
}

/// Output sequence length after the second conv (stride=2, same padding).
fn after_conv2() -> usize {
    conv1d_out_len(after_conv1(), CONV_KERNEL, 2, CONV_PADDING)
}

// ===========================================================================
// Test 1: Mel spectrogram processing (Conv1d -> GELU -> Conv1d)
// ===========================================================================

/// Build mel spectrogram feature extraction: Conv1d -> GELU -> Conv1d -> GELU.
///
/// Input: `[N_MEL, MEL_SEQ_LEN]` (Variable).
/// Output: `[D_MODEL, T_OUT]` where T_OUT = after_conv2().
fn build_mel_feature_extraction() -> (TensorKernelDef, usize) {
    let t_mid = after_conv1();
    let t_out = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_deep_mel_features");

    let mel = b.add_input("mel", &[N_MEL, MEL_SEQ_LEN]);

    // Conv stem #1: Conv1d(N_MEL -> D_MODEL, k=3, s=1, p=1) -> GELU
    let c1_w = b.add_input("conv1_w", &[D_MODEL, N_MEL, CONV_KERNEL]);
    let c1_b = b.add_input("conv1_b", &[D_MODEL]);
    let c1 = b.add_conv1d(mel, c1_w, Some(c1_b), 1, CONV_PADDING, &[D_MODEL, t_mid]);
    let g1 = b.add_gelu(c1, &[D_MODEL, t_mid]);

    // Conv stem #2: Conv1d(D_MODEL -> D_MODEL, k=3, s=2, p=1) -> GELU
    let c2_w = b.add_input("conv2_w", &[D_MODEL, D_MODEL, CONV_KERNEL]);
    let c2_b = b.add_input("conv2_b", &[D_MODEL]);
    let c2 = b.add_conv1d(g1, c2_w, Some(c2_b), 2, CONV_PADDING, &[D_MODEL, t_out]);
    let out = b.add_gelu(c2, &[D_MODEL, t_out]);

    (b.build(out).expect("valid mel feature kernel"), t_out)
}

fn mel_feature_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // mel
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, N_MEL, CONV_KERNEL]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, D_MODEL, CONV_KERNEL]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
    ]
}

#[test]
fn test_whisper_deep_mel_features_ibp() {
    let (def, t_out) = build_mel_feature_extraction();
    let bindings = mel_feature_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[D_MODEL, t_out]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Whisper deep mel features IBP: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
    assert!(lo < hi, "non-degenerate bounds");
}

#[test]
fn test_whisper_deep_mel_features_crown() {
    let (def, t_out) = build_mel_feature_extraction();
    let bindings = mel_feature_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 1.0);

    let (method, output, reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[D_MODEL, t_out]);
    eprintln!("Whisper deep mel features CROWN: method={method:?}");
    if let Some(r) = &reason {
        eprintln!("  fallback: {r}");
    }
}

// ===========================================================================
// Test 2: Encoder self-attention with sinusoidal PE
// ===========================================================================

/// Build encoder self-attention with pre-added sinusoidal positional encoding.
///
/// Input: `[T, D_MODEL]` (Variable -- after transpose + PE add).
/// Output: `[T, D_MODEL]`.
///
/// Architecture: LayerNorm -> MHA(standard) -> + residual
fn build_encoder_self_attn_with_pe() -> (TensorKernelDef, usize) {
    let t_out = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_deep_enc_sa_pe");

    // Input is the post-PE sequence [T, D_MODEL]
    let x = b.add_input("x", &[t_out, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);

    let shape = [t_out, D_MODEL];

    // Pre-norm -> MHA(standard) -> + residual
    let normed = b.add_layer_norm(x, eps, 1, ln_w, ln_b, &shape);
    let attn = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            out_w,
            N_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid encoder self-attention");
    let out = b.add_binary_add(x, attn, &shape);

    (b.build(out).expect("valid encoder SA+PE kernel"), t_out)
}

fn encoder_self_attn_pe_bindings() -> Vec<TensorParamBinding> {
    let d = D_MODEL;
    let t_out = after_conv2();
    let _ = t_out;
    let w = ArrayD::from_elem(IxDyn(&[d, d]), W_MAG);
    vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)), // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)), // ln_b
        TensorParamBinding::ConstantTensor(w.clone()), // q_w
        TensorParamBinding::ConstantTensor(w.clone()), // k_w
        TensorParamBinding::ConstantTensor(w.clone()), // v_w
        TensorParamBinding::ConstantTensor(w),    // out_w
    ]
}

#[test]
fn test_whisper_deep_enc_self_attn_pe_ibp() {
    let (def, t_out) = build_encoder_self_attn_with_pe();
    let bindings = encoder_self_attn_pe_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[t_out, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[t_out, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Whisper deep enc SA+PE IBP: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

#[test]
fn test_whisper_deep_enc_self_attn_pe_crown() {
    let (def, t_out) = build_encoder_self_attn_with_pe();
    let bindings = encoder_self_attn_pe_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[t_out, D_MODEL], 1.0);

    let (method, output, reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[t_out, D_MODEL]);
    eprintln!("Whisper deep enc SA+PE: method={method:?}");
    if let Some(r) = &reason {
        eprintln!("  fallback: {r}");
    }
}

// ===========================================================================
// Test 3: Encoder FFN (Linear -> GELU -> Linear with residual)
// ===========================================================================

/// Build standalone encoder FFN sub-block.
///
/// Input: `[T, D_MODEL]` (Variable).
/// Output: `[T, D_MODEL]`.
///
/// Architecture: LayerNorm -> Linear(D, 4D) -> GELU -> Linear(4D, D) -> + residual
fn build_encoder_ffn() -> (TensorKernelDef, usize) {
    let t = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_deep_enc_ffn");

    let x = b.add_input("x", &[t, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let shape = [t, D_MODEL];
    let ffn_shape = [t, FFN_DIM];

    let normed = b.add_layer_norm(x, eps, 1, ln_w, ln_b, &shape);
    let h = b.add_linear(normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(h, &ffn_shape);
    let proj = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(x, proj, &shape);

    (b.build(out).expect("valid encoder FFN kernel"), t)
}

fn encoder_ffn_bindings() -> Vec<TensorParamBinding> {
    let d = D_MODEL;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), W_MAG)),
    ]
}

#[test]
fn test_whisper_deep_enc_ffn_ibp() {
    let (def, t) = build_encoder_ffn();
    let bindings = encoder_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[t, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Whisper deep enc FFN IBP: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
    assert!(lo < hi);
}

#[test]
fn test_whisper_deep_enc_ffn_verify_record() {
    let (def, t) = build_encoder_ffn();
    let bindings = encoder_ffn_bindings();
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_deep_encoder_ffn");
    assert_eq!(result.num_variables, 1);
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t, D_MODEL]);

    // LayerNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Encoder FFN with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Test 4: Encoder layer (self-attn -> LN -> FFN -> LN)
// ===========================================================================

/// Build a full encoder layer: pre-norm self-attn + pre-norm FFN.
///
/// Input: `[T, D_MODEL]` (Variable).
/// Output: `[T, D_MODEL]`.
fn build_encoder_layer() -> (TensorKernelDef, usize) {
    let t = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_deep_enc_layer");

    let x = b.add_input("x", &[t, D_MODEL]);
    let eps = b.add_input("eps", &[1]);

    // Self-attention weights
    let sa_ln_w = b.add_input("sa_ln_w", &[D_MODEL]);
    let sa_ln_b = b.add_input("sa_ln_b", &[D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);

    // FFN weights
    let ffn_ln_w = b.add_input("ffn_ln_w", &[D_MODEL]);
    let ffn_ln_b = b.add_input("ffn_ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let shape = [t, D_MODEL];
    let ffn_shape = [t, FFN_DIM];

    // Sub-block 1: Pre-norm self-attention
    let sa_normed = b.add_layer_norm(x, eps, 1, sa_ln_w, sa_ln_b, &shape);
    let sa_out = b
        .add_multi_head_attention(
            sa_normed,
            q_w,
            k_w,
            v_w,
            out_w,
            N_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid self-attention");
    let r1 = b.add_binary_add(x, sa_out, &shape);

    // Sub-block 2: Pre-norm FFN
    let ffn_normed = b.add_layer_norm(r1, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
    let h = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(h, &ffn_shape);
    let proj = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(r1, proj, &shape);

    (b.build(out).expect("valid encoder layer kernel"), t)
}

fn encoder_layer_bindings() -> Vec<TensorParamBinding> {
    let d = D_MODEL;
    let w = ArrayD::from_elem(IxDyn(&[d, d]), W_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w),
        TensorParamBinding::ConstantTensor(ln_w),
        TensorParamBinding::ConstantTensor(ln_b),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), W_MAG)),
    ]
}

#[test]
fn test_whisper_deep_enc_layer_ibp() {
    let (def, t) = build_encoder_layer();
    let bindings = encoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[t, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Whisper deep enc layer IBP: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

#[test]
fn test_whisper_deep_enc_layer_verify_record() {
    let (def, t) = build_encoder_layer();
    let bindings = encoder_layer_bindings();
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_deep_encoder_layer");
    assert_eq!(result.num_variables, 1);

    // 2 LayerNorms => Heuristic
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
    );
}

// ===========================================================================
// Test 5: Decoder cross-attention (Q from decoder, K/V from encoder)
// ===========================================================================

/// Build decoder cross-attention with pre-norm LayerNorm.
///
/// Q input: `[DEC_SEQ, D_MODEL]` (Variable -- decoder hidden state).
/// KV input: `[ENC_SEQ, D_MODEL]` (Constant -- encoder output).
/// Output: `[DEC_SEQ, D_MODEL]`.
fn build_decoder_cross_attn() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_deep_dec_cross_attn");

    let q_in = b.add_input("dec_hidden", &[DEC_SEQ, D_MODEL]);
    let kv_in = b.add_input("enc_output", &[ENC_SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);

    let shape = [DEC_SEQ, D_MODEL];

    let normed = b.add_layer_norm(q_in, eps, 1, ln_w, ln_b, &shape);
    let attn = b
        .add_multi_head_cross_attention(
            normed,
            kv_in,
            q_w,
            k_w,
            v_w,
            out_w,
            N_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid cross-attention");
    let out = b.add_binary_add(q_in, attn, &shape);

    b.build(out).expect("valid decoder cross-attn kernel")
}

fn decoder_cross_attn_bindings() -> Vec<TensorParamBinding> {
    let d = D_MODEL;
    let w = ArrayD::from_elem(IxDyn(&[d, d]), W_MAG);
    vec![
        TensorParamBinding::Variable, // dec_hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_SEQ, d]), 0.1f32)), // enc_output
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w),
    ]
}

#[test]
fn test_whisper_deep_dec_cross_attn_ibp() {
    let def = build_decoder_cross_attn();
    let bindings = decoder_cross_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    // Output shape follows Q (decoder seq len), not KV (encoder seq len).
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Whisper deep dec cross-attn IBP: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

#[test]
fn test_whisper_deep_dec_cross_attn_crown() {
    let def = build_decoder_cross_attn();
    let bindings = decoder_cross_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, D_MODEL], 1.0);

    let (method, output, reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, D_MODEL]);
    eprintln!("Whisper deep dec cross-attn: method={method:?}");
    if let Some(r) = &reason {
        eprintln!("  fallback: {r}");
    }
}

// ===========================================================================
// Test 6: Decoder causal self-attention (causal mask)
// ===========================================================================

/// Build decoder causal self-attention.
///
/// Input: `[DEC_SEQ, D_MODEL]` (Variable).
/// Output: `[DEC_SEQ, D_MODEL]`.
///
/// Causal mask ensures each position only attends to previous positions,
/// which should produce tighter bounds than bidirectional attention.
fn build_decoder_causal_self_attn() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_deep_dec_causal_sa");

    let x = b.add_input("x", &[DEC_SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);

    let shape = [DEC_SEQ, D_MODEL];

    let normed = b.add_layer_norm(x, eps, 1, ln_w, ln_b, &shape);
    let attn = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            out_w,
            N_HEADS,
            AttentionMask::Causal, // causal, not standard
            &shape,
        )
        .expect("valid causal self-attention");
    let out = b.add_binary_add(x, attn, &shape);

    b.build(out).expect("valid decoder causal SA kernel")
}

fn decoder_causal_sa_bindings() -> Vec<TensorParamBinding> {
    let d = D_MODEL;
    let w = ArrayD::from_elem(IxDyn(&[d, d]), W_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w),
    ]
}

#[test]
fn test_whisper_deep_dec_causal_sa_ibp() {
    let def = build_decoder_causal_self_attn();
    let bindings = decoder_causal_sa_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Whisper deep dec causal SA IBP: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
    assert!(lo < hi);
}

#[test]
fn test_whisper_deep_dec_causal_sa_verify_record() {
    let def = build_decoder_causal_self_attn();
    let bindings = decoder_causal_sa_bindings();
    let input = uniform_bounds(&[DEC_SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "whisper_deep_decoder_causal_self_attn",
    );
    assert_eq!(result.num_variables, 1);

    // LayerNorm => Heuristic
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
    );
}

// ===========================================================================
// Test 7: Full encoder block -- 2-layer stacked encoder (IBP + CROWN)
// ===========================================================================

/// Build a 2-layer stacked encoder for bounds stability through depth.
///
/// Input: `[T, D_MODEL]` (Variable).
/// Output: `[T, D_MODEL]`.
///
/// Architecture: 2 x (LN -> MHA(standard) -> + res -> LN -> FFN -> + res)
fn build_2layer_encoder_stack() -> (TensorKernelDef, usize) {
    let t = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_deep_2layer_enc");

    let x = b.add_input("x", &[t, D_MODEL]);
    let eps = b.add_input("eps", &[1]);

    let shape = [t, D_MODEL];
    let ffn_shape = [t, FFN_DIM];
    let mut current = x;

    for idx in 0..2 {
        let pfx = format!("l{idx}");

        let sa_ln_w = b.add_input(&format!("{pfx}_sa_ln_w"), &[D_MODEL]);
        let sa_ln_b = b.add_input(&format!("{pfx}_sa_ln_b"), &[D_MODEL]);
        let q_w = b.add_input(&format!("{pfx}_qw"), &[D_MODEL, D_MODEL]);
        let k_w = b.add_input(&format!("{pfx}_kw"), &[D_MODEL, D_MODEL]);
        let v_w = b.add_input(&format!("{pfx}_vw"), &[D_MODEL, D_MODEL]);
        let out_w = b.add_input(&format!("{pfx}_ow"), &[D_MODEL, D_MODEL]);

        let ffn_ln_w = b.add_input(&format!("{pfx}_ffn_ln_w"), &[D_MODEL]);
        let ffn_ln_b = b.add_input(&format!("{pfx}_ffn_ln_b"), &[D_MODEL]);
        let ffn1_w = b.add_input(&format!("{pfx}_ffn1w"), &[FFN_DIM, D_MODEL]);
        let ffn2_w = b.add_input(&format!("{pfx}_ffn2w"), &[D_MODEL, FFN_DIM]);

        // Self-attention sub-block
        let sa_normed = b.add_layer_norm(current, eps, 1, sa_ln_w, sa_ln_b, &shape);
        let sa_out = b
            .add_multi_head_attention(
                sa_normed,
                q_w,
                k_w,
                v_w,
                out_w,
                N_HEADS,
                AttentionMask::Standard,
                &shape,
            )
            .expect("valid encoder self-attention");
        let r1 = b.add_binary_add(current, sa_out, &shape);

        // FFN sub-block
        let ffn_normed = b.add_layer_norm(r1, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
        let h = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
        let act = b.add_gelu(h, &ffn_shape);
        let proj = b.add_linear(act, ffn2_w, None, &shape);
        current = b.add_binary_add(r1, proj, &shape);
    }

    (b.build(current).expect("valid 2-layer encoder stack"), t)
}

fn two_layer_encoder_bindings() -> Vec<TensorParamBinding> {
    let d = D_MODEL;
    let w = ArrayD::from_elem(IxDyn(&[d, d]), W_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), W_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), W_MAG);

    let mut bindings = vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
    ];

    for _ in 0..2 {
        // Self-attention: ln_w, ln_b, q, k, v, out
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w.clone()));
        // FFN: ln_w, ln_b, ffn1_w, ffn2_w
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn1.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn2.clone()));
    }

    bindings
}

#[test]
fn test_whisper_deep_2layer_enc_ibp() {
    let (def, t) = build_2layer_encoder_stack();
    let bindings = two_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[t, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Whisper deep 2-layer enc IBP: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

#[test]
fn test_whisper_deep_2layer_enc_crown() {
    let (def, t) = build_2layer_encoder_stack();
    let bindings = two_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let (method, output, reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[t, D_MODEL]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Whisper deep 2-layer enc CROWN: method={method:?}, bounds=[{lo}, {hi}]");
    if let Some(r) = &reason {
        eprintln!("  fallback: {r}");
    }
}

#[test]
fn test_whisper_deep_2layer_enc_verify_record() {
    let (def, t) = build_2layer_encoder_stack();
    let bindings = two_layer_encoder_bindings();
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_deep_2layer_encoder_stack");
    assert_eq!(result.num_variables, 1);

    // 4 LayerNorms across 2 blocks => Heuristic
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
    );
}

// ===========================================================================
// Test 8: Token embedding + positional embedding
// ===========================================================================

/// Build token embedding + learned positional embedding.
///
/// Input: `[DEC_SEQ, D_MODEL]` (Variable -- continuous relaxation of tokens).
/// Output: `[DEC_SEQ, D_MODEL]`.
///
/// Architecture: token_emb + positional_emb (element-wise add).
/// This is the decoder entry point before transformer blocks.
fn build_token_pos_embedding() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_deep_tok_pos_emb");

    let tok_emb = b.add_input("tok_emb", &[DEC_SEQ, D_MODEL]);
    let pos_emb = b.add_input("pos_emb", &[DEC_SEQ, D_MODEL]);

    let out = b.add_binary_add(tok_emb, pos_emb, &[DEC_SEQ, D_MODEL]);

    b.build(out).expect("valid token+pos embedding kernel")
}

fn token_pos_embedding_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // tok_emb (Variable for verification)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DEC_SEQ, D_MODEL]), W_MAG)), // pos_emb (learned, constant)
    ]
}

#[test]
fn test_whisper_deep_token_pos_emb_ibp() {
    let def = build_token_pos_embedding();
    let bindings = token_pos_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, D_MODEL],);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Whisper deep tok+pos emb IBP: bounds=[{lo}, {hi}]");

    // Adding a small constant (W_MAG=0.02) to [-1, 1] => [-0.98, 1.02].
    // Bounds should be very tight -- no nonlinearities.
    assert!(lo.is_finite() && hi.is_finite());
    assert!(
        (hi - lo) < 5.0,
        "embedding addition bounds should be tight, got width {}",
        hi - lo
    );
}

#[test]
fn test_whisper_deep_token_pos_emb_verify_record() {
    let def = build_token_pos_embedding();
    let bindings = token_pos_embedding_bindings();
    let input = uniform_bounds(&[DEC_SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_deep_token_pos_embedding");
    assert_eq!(result.num_variables, 1);

    // Pure linear (add constant): should be Sound, not Heuristic.
    eprintln!(
        "Token+pos embedding soundness: {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Test 9: LM head (Linear -> softmax for token prediction)
// ===========================================================================

/// Build the LM head: Linear projection -> softmax.
///
/// Input: `[DEC_SEQ, D_MODEL]` (Variable -- decoder output after LayerNorm).
/// Output: `[DEC_SEQ, VOCAB_SIZE]`.
///
/// Architecture: LayerNorm -> matmul(D_MODEL, VOCAB_SIZE) -> softmax(dim=-1)
fn build_lm_head() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_deep_lm_head");

    let x = b.add_input("x", &[DEC_SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let proj_w = b.add_input("proj_w", &[D_MODEL, VOCAB_SIZE]);

    // Final LayerNorm
    let normed = b.add_layer_norm(x, eps, 1, ln_w, ln_b, &[DEC_SEQ, D_MODEL]);

    // Output projection: [DEC_SEQ, D_MODEL] x [D_MODEL, VOCAB_SIZE] -> [DEC_SEQ, VOCAB_SIZE]
    let logits = b.add_matmul(normed, proj_w, false, None, &[DEC_SEQ, VOCAB_SIZE]);

    // Softmax over last dimension (vocabulary)
    let probs = b.add_softmax(logits, -1, &[DEC_SEQ, VOCAB_SIZE]);

    b.build(probs).expect("valid LM head kernel")
}

fn lm_head_bindings() -> Vec<TensorParamBinding> {
    let d = D_MODEL;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, VOCAB_SIZE]), W_MAG)),
    ]
}

#[test]
fn test_whisper_deep_lm_head_ibp() {
    let def = build_lm_head();
    let bindings = lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, VOCAB_SIZE],);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Whisper deep LM head IBP: bounds=[{lo}, {hi}]");

    // Softmax output should be in [0, 1] but IBP bounds may be wider.
    // Check finiteness as the primary invariant.
    assert!(lo.is_finite() && hi.is_finite());
}

#[test]
fn test_whisper_deep_lm_head_verify_record() {
    let def = build_lm_head();
    let bindings = lm_head_bindings();
    let input = uniform_bounds(&[DEC_SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_deep_lm_head");
    assert_eq!(result.num_variables, 1);
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[DEC_SEQ, VOCAB_SIZE]);

    // LayerNorm + softmax => Heuristic
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
    );
}

// ===========================================================================
// Test 10: End-to-end encoder (Mel -> Conv -> N layers -> LayerNorm)
// ===========================================================================

/// Build the end-to-end Whisper encoder pipeline.
///
/// Input: `[N_MEL, MEL_SEQ_LEN]` (Variable -- mel spectrogram).
/// Output: `[T, D_MODEL]`.
///
/// Architecture:
///   Conv1d(N_MEL -> D, k=3, s=1, p=1) -> GELU ->
///   Conv1d(D -> D, k=3, s=2, p=1) -> GELU ->
///   Transpose([D, T] -> [T, D]) ->
///   + positional_embedding ->
///     2 x Encoder Block (LN + MHA(standard) + residual + LN + FFN + residual) ->
///     Final LayerNorm
fn build_e2e_encoder() -> (TensorKernelDef, usize) {
    let t_mid = after_conv1();
    let t_out = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_deep_e2e_encoder");

    // --- Mel input ---
    let mel = b.add_input("mel", &[N_MEL, MEL_SEQ_LEN]);

    // --- Conv stem #1 ---
    let c1_w = b.add_input("conv1_w", &[D_MODEL, N_MEL, CONV_KERNEL]);
    let c1_b = b.add_input("conv1_b", &[D_MODEL]);
    let c1 = b.add_conv1d(mel, c1_w, Some(c1_b), 1, CONV_PADDING, &[D_MODEL, t_mid]);
    let g1 = b.add_gelu(c1, &[D_MODEL, t_mid]);

    // --- Conv stem #2 ---
    let c2_w = b.add_input("conv2_w", &[D_MODEL, D_MODEL, CONV_KERNEL]);
    let c2_b = b.add_input("conv2_b", &[D_MODEL]);
    let c2 = b.add_conv1d(g1, c2_w, Some(c2_b), 2, CONV_PADDING, &[D_MODEL, t_out]);
    let g2 = b.add_gelu(c2, &[D_MODEL, t_out]);

    // --- Transpose + positional embedding ---
    let transposed = b.add_transpose(g2, &[1, 0], &[t_out, D_MODEL]);
    let pos_emb = b.add_input("pos_emb", &[t_out, D_MODEL]);
    let x = b.add_binary_add(transposed, pos_emb, &[t_out, D_MODEL]);

    // --- 2 encoder transformer blocks ---
    let eps = b.add_input("eps", &[1]);
    let shape = [t_out, D_MODEL];
    let ffn_shape = [t_out, FFN_DIM];
    let mut current = x;

    for idx in 0..2 {
        let pfx = format!("enc{idx}");

        let sa_ln_w = b.add_input(&format!("{pfx}_sa_ln_w"), &[D_MODEL]);
        let sa_ln_b = b.add_input(&format!("{pfx}_sa_ln_b"), &[D_MODEL]);
        let q_w = b.add_input(&format!("{pfx}_qw"), &[D_MODEL, D_MODEL]);
        let k_w = b.add_input(&format!("{pfx}_kw"), &[D_MODEL, D_MODEL]);
        let v_w = b.add_input(&format!("{pfx}_vw"), &[D_MODEL, D_MODEL]);
        let out_w = b.add_input(&format!("{pfx}_ow"), &[D_MODEL, D_MODEL]);

        let ffn_ln_w = b.add_input(&format!("{pfx}_ffn_ln_w"), &[D_MODEL]);
        let ffn_ln_b = b.add_input(&format!("{pfx}_ffn_ln_b"), &[D_MODEL]);
        let ffn1_w = b.add_input(&format!("{pfx}_ffn1w"), &[FFN_DIM, D_MODEL]);
        let ffn2_w = b.add_input(&format!("{pfx}_ffn2w"), &[D_MODEL, FFN_DIM]);

        let sa_normed = b.add_layer_norm(current, eps, 1, sa_ln_w, sa_ln_b, &shape);
        let sa_out = b
            .add_multi_head_attention(
                sa_normed,
                q_w,
                k_w,
                v_w,
                out_w,
                N_HEADS,
                AttentionMask::Standard,
                &shape,
            )
            .expect("valid encoder self-attention");
        let r1 = b.add_binary_add(current, sa_out, &shape);

        let ffn_normed = b.add_layer_norm(r1, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
        let h = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
        let act = b.add_gelu(h, &ffn_shape);
        let proj = b.add_linear(act, ffn2_w, None, &shape);
        current = b.add_binary_add(r1, proj, &shape);
    }

    // --- Final LayerNorm ---
    let final_ln_w = b.add_input("final_ln_w", &[D_MODEL]);
    let final_ln_b = b.add_input("final_ln_b", &[D_MODEL]);
    let final_eps = b.add_input("final_eps", &[1]);
    let output = b.add_layer_norm(current, final_eps, 1, final_ln_w, final_ln_b, &shape);

    (b.build(output).expect("valid e2e encoder kernel"), t_out)
}

fn e2e_encoder_bindings() -> Vec<TensorParamBinding> {
    let d = D_MODEL;
    let t_out = after_conv2();
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), W_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), W_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), W_MAG);

    let mut bindings = vec![
        TensorParamBinding::Variable, // mel
        // Conv stem #1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d, N_MEL, CONV_KERNEL]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
        // Conv stem #2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, d, CONV_KERNEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
        // Positional embedding
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[t_out, d]), W_MAG)),
        // Shared eps
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    // 2 encoder blocks
    for _ in 0..2 {
        // Self-attention: ln_w, ln_b, q, k, v, out
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        // FFN: ln_w, ln_b, ffn1_w, ffn2_w
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn1.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn2.clone()));
    }

    // Final LayerNorm
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // final_eps

    bindings
}

#[test]
fn test_whisper_deep_e2e_encoder_ibp() {
    let (def, t_out) = build_e2e_encoder();
    let bindings = e2e_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[t_out, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Whisper deep e2e encoder IBP: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

#[test]
fn test_whisper_deep_e2e_encoder_crown() {
    let (def, t_out) = build_e2e_encoder();
    let bindings = e2e_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 1.0);

    let (method, output, reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[t_out, D_MODEL]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Whisper deep e2e encoder CROWN: method={method:?}, bounds=[{lo}, {hi}]");
    if let Some(r) = &reason {
        eprintln!("  fallback: {r}");
    }
}

#[test]
fn test_whisper_deep_e2e_encoder_narrow_tighter() {
    let (def, _) = build_e2e_encoder();
    let bindings = e2e_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let wide = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 10.0);
    let narrow = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 1.0);

    let wide_out = graph.propagate_ibp(&wide).expect("wide IBP");
    let narrow_out = graph.propagate_ibp(&narrow).expect("narrow IBP");

    let (w_lo, w_hi) = wide_out.lower_upper();
    let (n_lo, n_hi) = narrow_out.lower_upper();

    let wide_range = w_hi.iter().zip(w_lo.iter()).map(|(h, l)| h - l);
    let narrow_range = n_hi.iter().zip(n_lo.iter()).map(|(h, l)| h - l);

    let tighter = wide_range.zip(narrow_range).filter(|(w, n)| n <= w).count();
    let total = w_lo.len();
    assert!(
        tighter > total / 2,
        "narrow input should produce tighter bounds for > 50% of elements, got {tighter}/{total}"
    );
}

#[test]
fn test_whisper_deep_e2e_encoder_verify_record() {
    let (def, t_out) = build_e2e_encoder();
    let bindings = e2e_encoder_bindings();
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_deep_e2e_encoder");
    assert_eq!(result.num_variables, 1);
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t_out, D_MODEL]);

    // 5 LayerNorms (2 per block + 1 final) => Heuristic
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
    );
}
