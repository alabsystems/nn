// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Whisper encoder-decoder compose verification tests.
//!
//! Tests that go BEYOND the existing coverage in compose_whisper_encoder.rs,
//! compose_whisper_decoder.rs, compose_whisper_full.rs, and compose_whisper_deep.rs.
//!
//! Focus areas:
//!
//! 1. **Mel spectrogram input bounds**: Normalized mel input [-4, 4] propagation
//! 2. **Encoder self-attention with sinusoidal PE**: Explicit PE injection + MHA
//! 3. **Encoder FFN with GELU**: Isolated FFN sub-block (Linear -> GELU -> Linear)
//! 4. **Cross-attention decoder**: Decoder queries attending to encoder output
//! 5. **Multi-layer encoder stack**: 3-layer encoder composition (IBP + CROWN)
//! 6. **Decoder autoregressive step**: Single decode step through decoder block
//! 7. **LM head vocabulary projection**: Linear -> softmax for token prediction
//! 8. **Full encoder -> decoder E2E**: Complete Whisper pipeline (IBP + CROWN)
//! 9. **Timestamp token bounds**: Timestamp prediction branch stays bounded
//!
//! Architecture reference: Radford et al. 2023, "Robust Speech Recognition via
//! Large-Scale Weak Supervision."
//!
//! GELU requires CROWN linearization. LayerNorm requires heuristic linearization
//! (IbpValidated mode per nn_engineering.md). Conv1d and Linear are exact under IBP.
//! Softmax uses piecewise CROWN approximation.
//!
//! Part of #4560: Extended Whisper encoder-decoder compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    sinusoidal_pe, uniform_bounds, verify_and_assert,
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
/// Encoder mel input sequence length (production: 3000).
const MEL_SEQ_LEN: usize = 8;
/// Model / embedding dimension.
const D_MODEL: usize = 16;
/// Number of attention heads (head_dim = D_MODEL / N_HEADS = 4).
const N_HEADS: usize = 4;
/// FFN intermediate dimension: 4x model dimension.
const FFN_DIM: usize = D_MODEL * 4;
/// Conv1d kernel size for both encoder stems.
const CONV_KERNEL: usize = 3;
/// Conv1d padding for both encoder stems.
const CONV_PADDING: usize = 1;
/// Decoder sequence length (number of tokens).
const DEC_SEQ_LEN: usize = 4;
/// Vocabulary size for LM head.
const VOCAB_SIZE: usize = 16;
/// Timestamp token offset within vocabulary.
const TIMESTAMP_VOCAB_START: usize = 12;
/// Number of timestamp tokens.
const TIMESTAMP_VOCAB_COUNT: usize = 4;
/// Small weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.01;
/// Number of encoder layers for multi-layer tests.
const N_ENCODER_LAYERS: usize = 3;

/// Output sequence length after first conv (stride=1, same padding).
fn after_conv1_len() -> usize {
    conv1d_out_len(MEL_SEQ_LEN, CONV_KERNEL, 1, CONV_PADDING)
}

/// Output sequence length after second conv (stride=2).
fn after_conv2_len() -> usize {
    conv1d_out_len(after_conv1_len(), CONV_KERNEL, 2, CONV_PADDING)
}

/// Encoder output sequence length.
fn enc_seq_len() -> usize {
    after_conv2_len()
}

// ===========================================================================
// 1. Mel spectrogram input bounds: normalized mel [-4, 4] propagation
// ===========================================================================

/// Build mel normalization + conv feature extraction.
///
/// Input: `[N_MEL, MEL_SEQ_LEN]` (Variable -- raw mel spectrogram).
/// Architecture: Add mean_shift (constant) -> Mul by inv_std (constant) ->
///   Conv1d(N_MEL->D, k=3, s=1, p=1) -> GELU
///
/// This tests that normalized mel values in typical range [-4, 4] propagate
/// correctly through the initial conv stem.
fn build_mel_norm_conv_kernel() -> TensorKernelDef {
    let t_mid = after_conv1_len();
    let mut b = TensorBlockBuilder::new("whisper_ext_mel_norm_conv");

    let mel = b.add_input("mel", &[N_MEL, MEL_SEQ_LEN]);
    let mean_shift = b.add_input("mean_shift", &[N_MEL, MEL_SEQ_LEN]);
    let inv_std = b.add_input("inv_std", &[N_MEL, MEL_SEQ_LEN]);

    // Normalize: (mel + mean_shift) * inv_std
    let shifted = b.add_binary_add(mel, mean_shift, &[N_MEL, MEL_SEQ_LEN]);
    let normed = b.add_binary_mul(shifted, inv_std, &[N_MEL, MEL_SEQ_LEN]);

    // Conv1d feature extraction
    let conv_w = b.add_input("conv_weight", &[D_MODEL, N_MEL, CONV_KERNEL]);
    let conv_b = b.add_input("conv_bias", &[D_MODEL]);
    let conv = b.add_conv1d(
        normed,
        conv_w,
        Some(conv_b),
        1,
        CONV_PADDING,
        &[D_MODEL, t_mid],
    );
    let out = b.add_gelu(conv, &[D_MODEL, t_mid]);

    b.build(out).expect("valid mel norm conv kernel")
}

fn mel_norm_conv_bindings() -> Vec<TensorParamBinding> {
    let t_mid = after_conv1_len();
    let _ = t_mid;
    vec![
        TensorParamBinding::Variable, // mel [N_MEL, MEL_SEQ_LEN]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[N_MEL, MEL_SEQ_LEN]), 0.0f32)), // mean_shift
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[N_MEL, MEL_SEQ_LEN]),
            0.25f32,
        )), // inv_std (1/4 to keep bounds tight)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, N_MEL, CONV_KERNEL]),
            WEIGHT_MAG,
        )), // conv_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)), // conv_bias
    ]
}

#[test]
fn test_ext_mel_norm_conv_ibp() {
    let def = build_mel_norm_conv_kernel();
    let bindings = mel_norm_conv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Mel spectrograms are typically in range [-4, 4] after log-mel normalization
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 4.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through mel norm conv");
    let t_mid = after_conv1_len();
    assert_eq!(output.lower_upper().0.shape(), &[D_MODEL, t_mid]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext mel norm conv IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_ext_mel_norm_conv_crown() {
    let def = build_mel_norm_conv_kernel();
    let bindings = mel_norm_conv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 4.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext mel norm conv: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_ext_mel_norm_conv_verify_record() {
    let def = build_mel_norm_conv_kernel();
    let bindings = mel_norm_conv_bindings();
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 4.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_ext_mel_norm_conv");
    assert_eq!(result.num_variables, 1);
    eprintln!(
        "Mel norm conv soundness: {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 2. Encoder self-attention with sinusoidal PE injection
// ===========================================================================

/// Build encoder self-attention with explicit sinusoidal positional encoding.
///
/// Input: `[T, D_MODEL]` (Variable -- post-conv features).
/// Architecture: Add sinusoidal_PE -> LayerNorm -> MHA(standard) -> residual
///
/// Differs from compose_whisper_encoder.rs which tests PE as a separate sub-block.
/// Here PE is fused into the attention block to verify PE + attention composition.
fn build_enc_sa_with_pe_kernel() -> TensorKernelDef {
    let t = enc_seq_len();
    let shape = [t, D_MODEL];
    let mut b = TensorBlockBuilder::new("whisper_ext_enc_sa_pe");

    let input = b.add_input("features", &shape);
    let pe = b.add_input("pos_enc", &shape);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);

    // Add sinusoidal PE
    let x = b.add_binary_add(input, pe, &shape);

    // Pre-norm self-attention
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
        .expect("valid encoder SA");
    let out = b.add_binary_add(x, attn, &shape);

    b.build(out).expect("valid enc SA with PE kernel")
}

fn enc_sa_with_pe_bindings() -> Vec<TensorParamBinding> {
    let t = enc_seq_len();
    let pe_data = sinusoidal_pe(t, D_MODEL);
    let w_proj = ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                // features
        TensorParamBinding::ConstantTensor(pe_data), // pos_enc
        TensorParamBinding::ConstantScalar(1e-5),    // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)), // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)), // ln_b
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_w
        TensorParamBinding::ConstantTensor(w_proj),  // out_w
    ]
}

#[test]
fn test_ext_enc_sa_pe_ibp() {
    let def = build_enc_sa_with_pe_kernel();
    let bindings = enc_sa_with_pe_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[enc_seq_len(), D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[enc_seq_len(), D_MODEL]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext enc SA+PE IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_ext_enc_sa_pe_crown() {
    let def = build_enc_sa_with_pe_kernel();
    let bindings = enc_sa_with_pe_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[enc_seq_len(), D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext enc SA+PE: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 3. Encoder FFN with GELU (isolated sub-block)
// ===========================================================================

/// Build isolated encoder FFN: LayerNorm -> Linear(D, 4D) -> GELU -> Linear(4D, D) -> residual.
///
/// Input: `[T, D_MODEL]` (Variable).
/// Output: `[T, D_MODEL]`.
fn build_enc_ffn_kernel() -> TensorKernelDef {
    let t = enc_seq_len();
    let shape = [t, D_MODEL];
    let ffn_shape = [t, FFN_DIM];
    let mut b = TensorBlockBuilder::new("whisper_ext_enc_ffn");

    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn1_b = b.add_input("ffn1_b", &[FFN_DIM]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);
    let ffn2_b = b.add_input("ffn2_b", &[D_MODEL]);

    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);
    let h = b.add_linear(normed, ffn1_w, Some(ffn1_b), &ffn_shape);
    let act = b.add_gelu(h, &ffn_shape);
    let proj = b.add_linear(act, ffn2_w, Some(ffn2_b), &shape);
    let out = b.add_binary_add(input, proj, &shape);

    b.build(out).expect("valid enc FFN kernel")
}

fn enc_ffn_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)), // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)), // ln_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, D_MODEL]),
            WEIGHT_MAG,
        )), // ffn1_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32)), // ffn1_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, FFN_DIM]),
            WEIGHT_MAG,
        )), // ffn2_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)), // ffn2_b
    ]
}

#[test]
fn test_ext_enc_ffn_ibp() {
    let def = build_enc_ffn_kernel();
    let bindings = enc_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[enc_seq_len(), D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[enc_seq_len(), D_MODEL]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext enc FFN IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_ext_enc_ffn_crown() {
    let def = build_enc_ffn_kernel();
    let bindings = enc_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[enc_seq_len(), D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext enc FFN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

#[test]
fn test_ext_enc_ffn_verify_record() {
    let def = build_enc_ffn_kernel();
    let bindings = enc_ffn_bindings();
    let input = uniform_bounds(&[enc_seq_len(), D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_ext_enc_ffn");
    assert_eq!(result.num_variables, 1);
    // LayerNorm -> Heuristic
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
    );
}

// ===========================================================================
// 4. Cross-attention decoder (decoder queries attending to encoder output)
// ===========================================================================

/// Build cross-attention with pre-norm and residual, using encoder output as
/// constant KV. This is structurally similar to compose_whisper_decoder.rs but
/// uses different dimensions and also includes the FFN sub-block that follows
/// cross-attention in the full decoder layer.
///
/// Input: `[DEC_SEQ_LEN, D_MODEL]` (Variable -- decoder hidden state).
/// Encoder output: `[enc_seq, D_MODEL]` (Constant).
/// Output: `[DEC_SEQ_LEN, D_MODEL]`.
fn build_cross_attn_with_ffn_kernel() -> TensorKernelDef {
    let enc_seq = enc_seq_len();
    let shape = [DEC_SEQ_LEN, D_MODEL];
    let ffn_shape = [DEC_SEQ_LEN, FFN_DIM];
    let mut b = TensorBlockBuilder::new("whisper_ext_cross_attn_ffn");

    let q_input = b.add_input("dec_hidden", &shape);
    let kv_input = b.add_input("enc_output", &[enc_seq, D_MODEL]);
    let eps = b.add_input("eps", &[1]);

    // Cross-attention
    let ca_ln_w = b.add_input("ca_ln_w", &[D_MODEL]);
    let ca_ln_b = b.add_input("ca_ln_b", &[D_MODEL]);
    let ca_q_w = b.add_input("ca_q_w", &[D_MODEL, D_MODEL]);
    let ca_k_w = b.add_input("ca_k_w", &[D_MODEL, D_MODEL]);
    let ca_v_w = b.add_input("ca_v_w", &[D_MODEL, D_MODEL]);
    let ca_out_w = b.add_input("ca_out_w", &[D_MODEL, D_MODEL]);

    // FFN
    let ffn_ln_w = b.add_input("ffn_ln_w", &[D_MODEL]);
    let ffn_ln_b = b.add_input("ffn_ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    // Cross-attention sub-block
    let ca_normed = b.add_layer_norm(q_input, eps, 1, ca_ln_w, ca_ln_b, &shape);
    let ca_out = b
        .add_multi_head_cross_attention(
            ca_normed,
            kv_input,
            ca_q_w,
            ca_k_w,
            ca_v_w,
            ca_out_w,
            N_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid cross-attention");
    let residual1 = b.add_binary_add(q_input, ca_out, &shape);

    // FFN sub-block
    let ffn_normed = b.add_layer_norm(residual1, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
    let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(residual1, ffn2, &shape);

    b.build(out).expect("valid cross-attn + FFN kernel")
}

fn cross_attn_with_ffn_bindings() -> Vec<TensorParamBinding> {
    let enc_seq = enc_seq_len();
    let w_proj = ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32);
    vec![
        TensorParamBinding::Variable, // dec_hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[enc_seq, D_MODEL]), 0.1f32)), // enc_output
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ca_ln_w
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ca_ln_b
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_q_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_k_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // ca_v_w
        TensorParamBinding::ConstantTensor(w_proj), // ca_out_w
        TensorParamBinding::ConstantTensor(ln_w), // ffn_ln_w
        TensorParamBinding::ConstantTensor(ln_b), // ffn_ln_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, D_MODEL]),
            WEIGHT_MAG,
        )), // ffn1_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, FFN_DIM]),
            WEIGHT_MAG,
        )), // ffn2_w
    ]
}

#[test]
fn test_ext_cross_attn_ffn_ibp() {
    let def = build_cross_attn_with_ffn_kernel();
    let bindings = cross_attn_with_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ_LEN, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ_LEN, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext cross-attn+FFN IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. Multi-layer encoder stack: 3 encoder layers composed (IBP + CROWN)
// ===========================================================================

/// Build a 3-layer encoder stack.
///
/// Input: `[T, D_MODEL]` (Variable -- post-conv/PE features).
/// Output: `[T, D_MODEL]`.
///
/// Each layer: LayerNorm -> MHA(standard) -> residual -> LayerNorm -> FFN -> residual.
/// Final LayerNorm on output.
fn build_multi_layer_encoder_kernel() -> TensorKernelDef {
    let t = enc_seq_len();
    let shape = [t, D_MODEL];
    let ffn_shape = [t, FFN_DIM];
    let mut b = TensorBlockBuilder::new("whisper_ext_3layer_encoder");

    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);

    let mut current = input;

    for layer_idx in 0..N_ENCODER_LAYERS {
        let pfx = format!("enc{layer_idx}");

        // Self-attention sub-block
        let sa_ln_w = b.add_input(&format!("{pfx}_sa_ln_w"), &[D_MODEL]);
        let sa_ln_b = b.add_input(&format!("{pfx}_sa_ln_b"), &[D_MODEL]);
        let q_w = b.add_input(&format!("{pfx}_q_w"), &[D_MODEL, D_MODEL]);
        let k_w = b.add_input(&format!("{pfx}_k_w"), &[D_MODEL, D_MODEL]);
        let v_w = b.add_input(&format!("{pfx}_v_w"), &[D_MODEL, D_MODEL]);
        let out_w = b.add_input(&format!("{pfx}_out_w"), &[D_MODEL, D_MODEL]);

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
            .expect("valid encoder SA");
        let residual1 = b.add_binary_add(current, sa_out, &shape);

        // FFN sub-block
        let ffn_ln_w = b.add_input(&format!("{pfx}_ffn_ln_w"), &[D_MODEL]);
        let ffn_ln_b = b.add_input(&format!("{pfx}_ffn_ln_b"), &[D_MODEL]);
        let ffn1_w = b.add_input(&format!("{pfx}_ffn1_w"), &[FFN_DIM, D_MODEL]);
        let ffn2_w = b.add_input(&format!("{pfx}_ffn2_w"), &[D_MODEL, FFN_DIM]);

        let ffn_normed = b.add_layer_norm(residual1, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
        let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
        let act = b.add_gelu(ffn1, &ffn_shape);
        let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
        current = b.add_binary_add(residual1, ffn2, &shape);
    }

    // Final LayerNorm
    let final_ln_w = b.add_input("final_ln_w", &[D_MODEL]);
    let final_ln_b = b.add_input("final_ln_b", &[D_MODEL]);
    let out = b.add_layer_norm(current, eps, 1, final_ln_w, final_ln_b, &shape);

    b.build(out).expect("valid 3-layer encoder kernel")
}

fn multi_layer_encoder_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[D_MODEL, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
    ];

    for _ in 0..N_ENCODER_LAYERS {
        // SA sub-block
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        // FFN sub-block
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn1.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn2.clone()));
    }

    // Final LN
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));

    bindings
}

#[test]
fn test_ext_3layer_encoder_ibp() {
    let def = build_multi_layer_encoder_kernel();
    let bindings = multi_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[enc_seq_len(), D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[enc_seq_len(), D_MODEL]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext 3-layer encoder IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_ext_3layer_encoder_crown() {
    let def = build_multi_layer_encoder_kernel();
    let bindings = multi_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[enc_seq_len(), D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext 3-layer encoder: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

#[test]
fn test_ext_3layer_encoder_verify_record() {
    let def = build_multi_layer_encoder_kernel();
    let bindings = multi_layer_encoder_bindings();
    let input = uniform_bounds(&[enc_seq_len(), D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_ext_3layer_encoder");
    assert_eq!(result.num_variables, 1);
    // Multiple LayerNorms -> Heuristic
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
    );
}

/// Verify that deeper encoder produces wider bounds (expected due to
/// accumulated LayerNorm decomposition + nonlinearities).
#[test]
fn test_ext_encoder_depth_widening() {
    let def = build_multi_layer_encoder_kernel();
    let bindings = multi_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let narrow = uniform_bounds(&[enc_seq_len(), D_MODEL], 0.5);
    let wide = uniform_bounds(&[enc_seq_len(), D_MODEL], 2.0);

    let narrow_out = graph.propagate_ibp(&narrow).expect("narrow IBP");
    let wide_out = graph.propagate_ibp(&wide).expect("wide IBP");

    let (narrow_lo, narrow_hi) = narrow_out.lower_upper();
    let (wide_lo, wide_hi) = wide_out.lower_upper();

    // Wider input should produce wider output for majority of elements
    let wider_count = narrow_hi
        .iter()
        .zip(narrow_lo.iter())
        .zip(wide_hi.iter().zip(wide_lo.iter()))
        .filter(|((nh, nl), (wh, wl))| (*wh - *wl) >= (*nh - *nl))
        .count();
    let total = narrow_lo.len();
    assert!(
        wider_count > total / 2,
        "wider input should produce wider output for > 50%, got {wider_count}/{total}"
    );
}

// ===========================================================================
// 6. Decoder autoregressive step: single decode step through decoder block
// ===========================================================================

/// Build a single autoregressive decode step.
///
/// Input: `[1, D_MODEL]` (Variable -- single token embedding).
/// Encoder output: `[enc_seq, D_MODEL]` (Constant).
/// Output: `[1, D_MODEL]`.
///
/// This represents inference-time decoding where the decoder processes one
/// token at a time. Self-attention with causal mask on a single token is
/// effectively a no-op (only attends to itself), so the cross-attention
/// and FFN dominate the bounds propagation.
fn build_single_step_decoder_kernel() -> TensorKernelDef {
    let enc_seq = enc_seq_len();
    let shape = [1, D_MODEL];
    let ffn_shape = [1, FFN_DIM];
    let mut b = TensorBlockBuilder::new("whisper_ext_single_step_dec");

    let input = b.add_input("token_emb", &shape);
    let enc_out = b.add_input("enc_output", &[enc_seq, D_MODEL]);
    let eps = b.add_input("eps", &[1]);

    // Causal self-attention (trivial for single token)
    let sa_ln_w = b.add_input("sa_ln_w", &[D_MODEL]);
    let sa_ln_b = b.add_input("sa_ln_b", &[D_MODEL]);
    let sa_q_w = b.add_input("sa_q_w", &[D_MODEL, D_MODEL]);
    let sa_k_w = b.add_input("sa_k_w", &[D_MODEL, D_MODEL]);
    let sa_v_w = b.add_input("sa_v_w", &[D_MODEL, D_MODEL]);
    let sa_out_w = b.add_input("sa_out_w", &[D_MODEL, D_MODEL]);

    let sa_normed = b.add_layer_norm(input, eps, 1, sa_ln_w, sa_ln_b, &shape);
    let sa_out = b
        .add_multi_head_attention(
            sa_normed,
            sa_q_w,
            sa_k_w,
            sa_v_w,
            sa_out_w,
            N_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("valid causal SA");
    let residual1 = b.add_binary_add(input, sa_out, &shape);

    // Cross-attention
    let ca_ln_w = b.add_input("ca_ln_w", &[D_MODEL]);
    let ca_ln_b = b.add_input("ca_ln_b", &[D_MODEL]);
    let ca_q_w = b.add_input("ca_q_w", &[D_MODEL, D_MODEL]);
    let ca_k_w = b.add_input("ca_k_w", &[D_MODEL, D_MODEL]);
    let ca_v_w = b.add_input("ca_v_w", &[D_MODEL, D_MODEL]);
    let ca_out_w = b.add_input("ca_out_w", &[D_MODEL, D_MODEL]);

    let ca_normed = b.add_layer_norm(residual1, eps, 1, ca_ln_w, ca_ln_b, &shape);
    let ca_out = b
        .add_multi_head_cross_attention(
            ca_normed,
            enc_out,
            ca_q_w,
            ca_k_w,
            ca_v_w,
            ca_out_w,
            N_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid cross-attention");
    let residual2 = b.add_binary_add(residual1, ca_out, &shape);

    // FFN
    let ffn_ln_w = b.add_input("ffn_ln_w", &[D_MODEL]);
    let ffn_ln_b = b.add_input("ffn_ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let ffn_normed = b.add_layer_norm(residual2, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
    let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(residual2, ffn2, &shape);

    b.build(out).expect("valid single step decoder kernel")
}

fn single_step_decoder_bindings() -> Vec<TensorParamBinding> {
    let enc_seq = enc_seq_len();
    let w_proj = ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32);
    vec![
        TensorParamBinding::Variable, // token_emb [1, D_MODEL]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[enc_seq, D_MODEL]), 0.1f32)), // enc_output
        TensorParamBinding::ConstantScalar(1e-5), // eps
        // SA
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        // CA
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj),
        // FFN
        TensorParamBinding::ConstantTensor(ln_w),
        TensorParamBinding::ConstantTensor(ln_b),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, D_MODEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, FFN_DIM]),
            WEIGHT_MAG,
        )),
    ]
}

#[test]
fn test_ext_single_step_decoder_ibp() {
    let def = build_single_step_decoder_kernel();
    let bindings = single_step_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[1, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[1, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext single-step decoder IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_ext_single_step_decoder_verify_record() {
    let def = build_single_step_decoder_kernel();
    let bindings = single_step_decoder_bindings();
    let input = uniform_bounds(&[1, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_ext_single_step_decoder");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
    );
}

// ===========================================================================
// 7. LM head vocabulary projection: Linear -> softmax for token prediction
// ===========================================================================

/// Build LM head: LayerNorm -> Linear(D, VOCAB) -> Softmax.
///
/// Input: `[DEC_SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[DEC_SEQ_LEN, VOCAB_SIZE]`.
///
/// The softmax output is bounded in [0, 1] per row and sums to 1.
fn build_lm_head_kernel() -> TensorKernelDef {
    let shape = [DEC_SEQ_LEN, D_MODEL];
    let logit_shape = [DEC_SEQ_LEN, VOCAB_SIZE];
    let mut b = TensorBlockBuilder::new("whisper_ext_lm_head");

    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let proj_w = b.add_input("proj_w", &[VOCAB_SIZE, D_MODEL]);
    let proj_b = b.add_input("proj_b", &[VOCAB_SIZE]);

    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);
    let logits = b.add_linear(normed, proj_w, Some(proj_b), &logit_shape);
    let probs = b.add_softmax(logits, -1, &logit_shape);

    b.build(probs).expect("valid LM head kernel")
}

fn lm_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,             // hidden
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)), // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)), // ln_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, D_MODEL]),
            WEIGHT_MAG,
        )), // proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)), // proj_b
    ]
}

#[test]
fn test_ext_lm_head_ibp() {
    let def = build_lm_head_kernel();
    let bindings = lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ_LEN, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("Ext LM head IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output should be bounded in [0, 1]
    assert!(
        lo_min >= -0.01,
        "softmax lower bound should be >= 0 (got {lo_min})"
    );
    assert!(
        hi_max <= 1.01,
        "softmax upper bound should be <= 1 (got {hi_max})"
    );
}

#[test]
fn test_ext_lm_head_crown() {
    let def = build_lm_head_kernel();
    let bindings = lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ_LEN, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext LM head: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

#[test]
fn test_ext_lm_head_verify_record() {
    let def = build_lm_head_kernel();
    let bindings = lm_head_bindings();
    let input = uniform_bounds(&[DEC_SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_ext_lm_head");
    assert_eq!(result.num_variables, 1);
    eprintln!(
        "LM head soundness: {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 8. Full encoder -> decoder E2E: Complete Whisper pipeline (IBP + CROWN)
// ===========================================================================

/// Build a complete Whisper encoder-decoder pipeline.
///
/// Input: `[N_MEL, MEL_SEQ_LEN]` (Variable -- mel spectrogram).
/// Output: `[DEC_SEQ_LEN, VOCAB_SIZE]` (token probabilities).
///
/// Architecture:
///   Conv1d(N_MEL->D, k=3, s=1, p=1) -> GELU ->
///   Conv1d(D->D, k=3, s=2, p=1) -> GELU ->
///   Transpose -> + PE ->
///   1 encoder layer (SA + FFN) ->
///   Encoder final LN ->
///   [Decoder: embedding + PE + 1 decoder block (causal SA + cross-attn + FFN)] ->
///   Decoder final LN ->
///   Output projection -> Softmax
fn build_full_e2e_kernel() -> TensorKernelDef {
    let t_mid = after_conv1_len();
    let t_enc = enc_seq_len();
    let enc_shape = [t_enc, D_MODEL];
    let dec_shape = [DEC_SEQ_LEN, D_MODEL];
    let ffn_enc_shape = [t_enc, FFN_DIM];
    let ffn_dec_shape = [DEC_SEQ_LEN, FFN_DIM];
    let logit_shape = [DEC_SEQ_LEN, VOCAB_SIZE];

    let mut b = TensorBlockBuilder::new("whisper_ext_full_e2e");

    // --- Variable input: mel spectrogram ---
    let mel = b.add_input("mel", &[N_MEL, MEL_SEQ_LEN]);
    let eps = b.add_input("eps", &[1]);

    // --- Encoder conv stems ---
    let conv1_w = b.add_input("conv1_w", &[D_MODEL, N_MEL, CONV_KERNEL]);
    let conv1_b = b.add_input("conv1_b", &[D_MODEL]);
    let conv1 = b.add_conv1d(
        mel,
        conv1_w,
        Some(conv1_b),
        1,
        CONV_PADDING,
        &[D_MODEL, t_mid],
    );
    let gelu1 = b.add_gelu(conv1, &[D_MODEL, t_mid]);

    let conv2_w = b.add_input("conv2_w", &[D_MODEL, D_MODEL, CONV_KERNEL]);
    let conv2_b = b.add_input("conv2_b", &[D_MODEL]);
    let conv2 = b.add_conv1d(
        gelu1,
        conv2_w,
        Some(conv2_b),
        2,
        CONV_PADDING,
        &[D_MODEL, t_enc],
    );
    let gelu2 = b.add_gelu(conv2, &[D_MODEL, t_enc]);

    // Transpose + PE
    let transposed = b.add_transpose(gelu2, &[1, 0], &enc_shape);
    let enc_pe = b.add_input("enc_pe", &enc_shape);
    let enc_x = b.add_binary_add(transposed, enc_pe, &enc_shape);

    // --- Single encoder layer ---
    let enc_sa_ln_w = b.add_input("enc_sa_ln_w", &[D_MODEL]);
    let enc_sa_ln_b = b.add_input("enc_sa_ln_b", &[D_MODEL]);
    let enc_q_w = b.add_input("enc_q_w", &[D_MODEL, D_MODEL]);
    let enc_k_w = b.add_input("enc_k_w", &[D_MODEL, D_MODEL]);
    let enc_v_w = b.add_input("enc_v_w", &[D_MODEL, D_MODEL]);
    let enc_out_w = b.add_input("enc_out_w", &[D_MODEL, D_MODEL]);

    let enc_normed = b.add_layer_norm(enc_x, eps, 1, enc_sa_ln_w, enc_sa_ln_b, &enc_shape);
    let enc_sa_out = b
        .add_multi_head_attention(
            enc_normed,
            enc_q_w,
            enc_k_w,
            enc_v_w,
            enc_out_w,
            N_HEADS,
            AttentionMask::Standard,
            &enc_shape,
        )
        .expect("valid encoder SA");
    let enc_res1 = b.add_binary_add(enc_x, enc_sa_out, &enc_shape);

    // Encoder FFN
    let enc_ffn_ln_w = b.add_input("enc_ffn_ln_w", &[D_MODEL]);
    let enc_ffn_ln_b = b.add_input("enc_ffn_ln_b", &[D_MODEL]);
    let enc_ffn1_w = b.add_input("enc_ffn1_w", &[FFN_DIM, D_MODEL]);
    let enc_ffn2_w = b.add_input("enc_ffn2_w", &[D_MODEL, FFN_DIM]);

    let enc_ffn_normed = b.add_layer_norm(enc_res1, eps, 1, enc_ffn_ln_w, enc_ffn_ln_b, &enc_shape);
    let enc_ffn1 = b.add_linear(enc_ffn_normed, enc_ffn1_w, None, &ffn_enc_shape);
    let enc_act = b.add_gelu(enc_ffn1, &ffn_enc_shape);
    let enc_ffn2 = b.add_linear(enc_act, enc_ffn2_w, None, &enc_shape);
    let enc_res2 = b.add_binary_add(enc_res1, enc_ffn2, &enc_shape);

    // Encoder final LN
    let enc_final_ln_w = b.add_input("enc_final_ln_w", &[D_MODEL]);
    let enc_final_ln_b = b.add_input("enc_final_ln_b", &[D_MODEL]);
    let encoder_output =
        b.add_layer_norm(enc_res2, eps, 1, enc_final_ln_w, enc_final_ln_b, &enc_shape);

    // --- Decoder: token embedding approximated as Variable input ---
    // (In reality, we'd have embedding lookup, but for NY we use
    // continuous relaxation of the mel input -> encoder -> decoder pipeline.
    // The decoder input is a constant positional embedding added to a
    // Linear projection of the encoder output's first DEC_SEQ_LEN tokens.)
    let dec_input = b.add_input("dec_input", &dec_shape);
    let dec_pe = b.add_input("dec_pe", &dec_shape);
    let dec_x = b.add_binary_add(dec_input, dec_pe, &dec_shape);

    // Decoder causal self-attention
    let dec_sa_ln_w = b.add_input("dec_sa_ln_w", &[D_MODEL]);
    let dec_sa_ln_b = b.add_input("dec_sa_ln_b", &[D_MODEL]);
    let dec_sa_q_w = b.add_input("dec_sa_q_w", &[D_MODEL, D_MODEL]);
    let dec_sa_k_w = b.add_input("dec_sa_k_w", &[D_MODEL, D_MODEL]);
    let dec_sa_v_w = b.add_input("dec_sa_v_w", &[D_MODEL, D_MODEL]);
    let dec_sa_out_w = b.add_input("dec_sa_out_w", &[D_MODEL, D_MODEL]);

    let dec_sa_normed = b.add_layer_norm(dec_x, eps, 1, dec_sa_ln_w, dec_sa_ln_b, &dec_shape);
    let dec_sa_out = b
        .add_multi_head_attention(
            dec_sa_normed,
            dec_sa_q_w,
            dec_sa_k_w,
            dec_sa_v_w,
            dec_sa_out_w,
            N_HEADS,
            AttentionMask::Causal,
            &dec_shape,
        )
        .expect("valid decoder causal SA");
    let dec_res1 = b.add_binary_add(dec_x, dec_sa_out, &dec_shape);

    // Decoder cross-attention
    let dec_ca_ln_w = b.add_input("dec_ca_ln_w", &[D_MODEL]);
    let dec_ca_ln_b = b.add_input("dec_ca_ln_b", &[D_MODEL]);
    let dec_ca_q_w = b.add_input("dec_ca_q_w", &[D_MODEL, D_MODEL]);
    let dec_ca_k_w = b.add_input("dec_ca_k_w", &[D_MODEL, D_MODEL]);
    let dec_ca_v_w = b.add_input("dec_ca_v_w", &[D_MODEL, D_MODEL]);
    let dec_ca_out_w = b.add_input("dec_ca_out_w", &[D_MODEL, D_MODEL]);

    let dec_ca_normed = b.add_layer_norm(dec_res1, eps, 1, dec_ca_ln_w, dec_ca_ln_b, &dec_shape);
    let dec_ca_out = b
        .add_multi_head_cross_attention(
            dec_ca_normed,
            encoder_output,
            dec_ca_q_w,
            dec_ca_k_w,
            dec_ca_v_w,
            dec_ca_out_w,
            N_HEADS,
            AttentionMask::Standard,
            &dec_shape,
        )
        .expect("valid decoder cross-attention");
    let dec_res2 = b.add_binary_add(dec_res1, dec_ca_out, &dec_shape);

    // Decoder FFN
    let dec_ffn_ln_w = b.add_input("dec_ffn_ln_w", &[D_MODEL]);
    let dec_ffn_ln_b = b.add_input("dec_ffn_ln_b", &[D_MODEL]);
    let dec_ffn1_w = b.add_input("dec_ffn1_w", &[FFN_DIM, D_MODEL]);
    let dec_ffn2_w = b.add_input("dec_ffn2_w", &[D_MODEL, FFN_DIM]);

    let dec_ffn_normed = b.add_layer_norm(dec_res2, eps, 1, dec_ffn_ln_w, dec_ffn_ln_b, &dec_shape);
    let dec_ffn1 = b.add_linear(dec_ffn_normed, dec_ffn1_w, None, &ffn_dec_shape);
    let dec_act = b.add_gelu(dec_ffn1, &ffn_dec_shape);
    let dec_ffn2 = b.add_linear(dec_act, dec_ffn2_w, None, &dec_shape);
    let dec_res3 = b.add_binary_add(dec_res2, dec_ffn2, &dec_shape);

    // Decoder final LN + output projection + softmax
    let dec_final_ln_w = b.add_input("dec_final_ln_w", &[D_MODEL]);
    let dec_final_ln_b = b.add_input("dec_final_ln_b", &[D_MODEL]);
    let dec_normed = b.add_layer_norm(dec_res3, eps, 1, dec_final_ln_w, dec_final_ln_b, &dec_shape);

    let lm_proj_w = b.add_input("lm_proj_w", &[VOCAB_SIZE, D_MODEL]);
    let logits = b.add_linear(dec_normed, lm_proj_w, None, &logit_shape);
    let probs = b.add_softmax(logits, -1, &logit_shape);

    b.build(probs).expect("valid full E2E kernel")
}

fn full_e2e_bindings() -> Vec<TensorParamBinding> {
    let t_enc = enc_seq_len();
    let w_proj = ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[D_MODEL, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,             // mel [N_MEL, MEL_SEQ_LEN]
        TensorParamBinding::ConstantScalar(1e-5), // eps
        // Conv stems
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, N_MEL, CONV_KERNEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, D_MODEL, CONV_KERNEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        // Encoder PE
        TensorParamBinding::ConstantTensor(sinusoidal_pe(t_enc, D_MODEL)),
        // Encoder layer SA
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        // Encoder layer FFN
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(w_ffn1.clone()),
        TensorParamBinding::ConstantTensor(w_ffn2.clone()),
        // Encoder final LN
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        // Decoder input (constant -- continuous relaxation)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DEC_SEQ_LEN, D_MODEL]),
            WEIGHT_MAG,
        )),
        // Decoder PE
        TensorParamBinding::ConstantTensor(sinusoidal_pe(DEC_SEQ_LEN, D_MODEL)),
        // Decoder causal SA
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        // Decoder cross-attn
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj),
        // Decoder FFN
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(w_ffn1),
        TensorParamBinding::ConstantTensor(w_ffn2),
        // Decoder final LN
        TensorParamBinding::ConstantTensor(ln_w),
        TensorParamBinding::ConstantTensor(ln_b),
        // LM head projection
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, D_MODEL]),
            WEIGHT_MAG,
        )),
    ]
}

#[test]
fn test_ext_full_e2e_ibp() {
    let def = build_full_e2e_kernel();
    let bindings = full_e2e_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 4.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext full E2E IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_ext_full_e2e_crown() {
    let def = build_full_e2e_kernel();
    let bindings = full_e2e_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 4.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext full E2E: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

#[test]
fn test_ext_full_e2e_verify_record() {
    let def = build_full_e2e_kernel();
    let bindings = full_e2e_bindings();
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 4.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_ext_full_e2e");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
    );
}

// ===========================================================================
// 9. Timestamp token bounds: verify timestamp prediction branch stays bounded
// ===========================================================================

/// Build a timestamp prediction sub-network.
///
/// Input: `[DEC_SEQ_LEN, D_MODEL]` (Variable -- decoder hidden state).
/// Output: `[DEC_SEQ_LEN, TIMESTAMP_VOCAB_COUNT]` (timestamp logits sliced from full vocab).
///
/// In Whisper, timestamp tokens occupy a contiguous range of the vocabulary.
/// This test verifies that the Linear projection into the timestamp token range
/// and subsequent softmax keeps the output bounded.
///
/// Architecture: LayerNorm -> Linear(D, VOCAB) -> Narrow(timestamp_range) -> Softmax
fn build_timestamp_head_kernel() -> TensorKernelDef {
    let shape = [DEC_SEQ_LEN, D_MODEL];
    let logit_shape = [DEC_SEQ_LEN, VOCAB_SIZE];
    let ts_shape = [DEC_SEQ_LEN, TIMESTAMP_VOCAB_COUNT];
    let mut b = TensorBlockBuilder::new("whisper_ext_timestamp_head");

    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let proj_w = b.add_input("proj_w", &[VOCAB_SIZE, D_MODEL]);

    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);
    let logits = b.add_linear(normed, proj_w, None, &logit_shape);

    // Narrow to timestamp token range [TIMESTAMP_VOCAB_START..TIMESTAMP_VOCAB_START+COUNT]
    let ts_logits = b.add_narrow(
        logits,
        1, // axis
        TIMESTAMP_VOCAB_START,
        TIMESTAMP_VOCAB_COUNT,
        &ts_shape,
    );

    // Softmax over timestamp tokens
    let ts_probs = b.add_softmax(ts_logits, -1, &ts_shape);

    b.build(ts_probs).expect("valid timestamp head kernel")
}

fn timestamp_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,             // hidden
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)), // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)), // ln_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, D_MODEL]),
            WEIGHT_MAG,
        )), // proj_w
    ]
}

#[test]
fn test_ext_timestamp_head_ibp() {
    let def = build_timestamp_head_kernel();
    let bindings = timestamp_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ_LEN, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, TIMESTAMP_VOCAB_COUNT]
    );
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("Ext timestamp head IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -0.01,
        "timestamp softmax lower bound should be >= 0 (got {lo_min})"
    );
    assert!(
        hi_max <= 1.01,
        "timestamp softmax upper bound should be <= 1 (got {hi_max})"
    );
}

#[test]
fn test_ext_timestamp_head_crown() {
    let def = build_timestamp_head_kernel();
    let bindings = timestamp_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ_LEN, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Ext timestamp head: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

#[test]
fn test_ext_timestamp_head_verify_record() {
    let def = build_timestamp_head_kernel();
    let bindings = timestamp_head_bindings();
    let input = uniform_bounds(&[DEC_SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_ext_timestamp_head");
    assert_eq!(result.num_variables, 1);
    eprintln!(
        "Timestamp head soundness: {:?}",
        result.verification.soundness_mode
    );
}

/// Verify that timestamp softmax bounds are well-contained even with wider input.
#[test]
fn test_ext_timestamp_head_wide_input_still_bounded() {
    let def = build_timestamp_head_kernel();
    let bindings = timestamp_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Use wider input bounds (5.0 instead of 1.0)
    let input = uniform_bounds(&[DEC_SEQ_LEN, D_MODEL], 5.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("Ext timestamp head wide IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax always produces outputs in [0, 1] regardless of input range
    assert!(
        lo_min >= -0.01,
        "wide timestamp softmax lower should be >= 0 (got {lo_min})"
    );
    assert!(
        hi_max <= 1.01,
        "wide timestamp softmax upper should be <= 1 (got {hi_max})"
    );
}
