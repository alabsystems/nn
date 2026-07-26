// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for Silero VAD + Whisper audio processing pipeline bounds.
//!
//! Verifies IBP and CROWN bound propagation through the audio processing
//! pipeline used alongside dpdf for audio-visual document understanding:
//!
//! ## Tests (18 tests)
//!
//! 1.  **Mel spectrogram feature extraction bounds** (IBP)
//! 2.  **VAD speech/silence probability bounds [0,1]** (IBP)
//! 3.  **Whisper encoder self-attention bounds** (IBP + CROWN)
//! 4.  **Whisper decoder cross-attention bounds** (IBP + CROWN)
//! 5.  **Log-mel feature normalization bounds** (IBP)
//! 6.  **Token probability output bounds (softmax)** (IBP)
//! 7.  **Full VAD-to-Whisper pipeline composition** (IBP)
//! 8.  **Audio preprocessing (resampling) bounds** (IBP)
//! 9.  **Whisper encoder layer norm bounds** (IBP)
//! 10. **Whisper decoder causal mask bounds** (IBP + CROWN)
//! 11. **Positional encoding for audio frames** (IBP)
//! 12. **Multi-head attention output bounds** (IBP + CROWN)
//! 13. **Feed-forward network intermediate bounds** (IBP + CROWN)
//! 14. **KV cache bounds in decoder** (IBP)
//! 15. **Beam search score bounds** (IBP)
//! 16. **Audio energy / volume bounds** (IBP)
//! 17. **Speech segment concatenation bounds** (IBP)
//! 18. **CTC decoder output bounds** (IBP)
//!
//! Architecture references:
//! - Silero VAD: Conv1d-based voice activity detector, sigmoid output [0, 1]
//! - Whisper (Radford et al., 2022): Encoder-decoder transformer for STT
//! - Mel spectrogram: STFT -> power -> mel filterbank projection
//! - Production dims: Whisper d_model=512, heads=8, FFN=2048, mel_bins=80
//!
//! Dimensions (small for fast verification, structurally representative):
//! - AUDIO_LEN=64 (raw audio samples), MEL_BINS=16, FRAMES=8
//! - D_MODEL=32, FFN_DIM=64, NUM_HEADS=4, HEAD_DIM=8
//! - VOCAB_SIZE=10, MAX_DECODE_LEN=8
//!
//! Part of #4204: Compose tests for audio processing pipeline.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// of Silero VAD + Whisper (production: d_model=512, heads=8, FFN=2048)
// ---------------------------------------------------------------------------

const AUDIO_LEN: usize = 64;
const MEL_BINS: usize = 16;
const FRAMES: usize = 8;
const D_MODEL: usize = 32;
const FFN_DIM: usize = 64;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = D_MODEL / NUM_HEADS; // 8
const VOCAB_SIZE: usize = 10;
const MAX_DECODE_LEN: usize = 8;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Zero bias tensor binding.
fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Ones tensor binding (for LayerNorm weight).
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Scalar epsilon binding.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Audio-domain input bounds: waveform samples in [-1, 1].
fn audio_bounds(len: usize) -> BoundedTensor {
    uniform_bounds(&[len], 1.0)
}

/// Sequence-domain input bounds: embeddings in [-range, +range].
fn seq_bounds(seq_len: usize, dim: usize, range: f32) -> BoundedTensor {
    uniform_bounds(&[seq_len, dim], range)
}

/// Add a single Whisper-style encoder block (pre-norm transformer) to a builder.
///
/// LN -> MHA -> residual -> LN -> FFN(GELU) -> residual.
/// Input/output: [SEQ, DIM]. Returns output node.
fn add_encoder_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    seq_len: usize,
    dim: usize,
    ffn_dim: usize,
    num_heads: usize,
    prefix: &str,
) -> TensorNodeId {
    let shape = [seq_len, dim];
    let ffn_shape = [seq_len, ffn_dim];

    // Pre-norm 1: LayerNorm
    let ln1_w = b.add_input(&format!("{prefix}_ln1_w"), &[dim]);
    let ln1_b = b.add_input(&format!("{prefix}_ln1_b"), &[dim]);
    let eps = b.add_input(&format!("{prefix}_ln1_eps"), &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln1_w, ln1_b, &shape);

    // Multi-head self-attention
    let qw = b.add_input(&format!("{prefix}_q_w"), &[dim, dim]);
    let kw = b.add_input(&format!("{prefix}_k_w"), &[dim, dim]);
    let vw = b.add_input(&format!("{prefix}_v_w"), &[dim, dim]);
    let ow = b.add_input(&format!("{prefix}_o_w"), &[dim, dim]);
    let attn = b
        .add_multi_head_attention(
            normed,
            qw,
            kw,
            vw,
            ow,
            num_heads,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // Residual 1
    let res1 = b.add_binary_add(input, attn, &shape);

    // Pre-norm 2: LayerNorm
    let ln2_w = b.add_input(&format!("{prefix}_ln2_w"), &[dim]);
    let ln2_b = b.add_input(&format!("{prefix}_ln2_b"), &[dim]);
    let eps2 = b.add_input(&format!("{prefix}_ln2_eps"), &[1]);
    let normed2 = b.add_layer_norm(res1, eps2, 1, ln2_w, ln2_b, &shape);

    // FFN: Linear -> GELU -> Linear
    let ffn1_w = b.add_input(&format!("{prefix}_ffn1_w"), &[ffn_dim, dim]);
    let ffn2_w = b.add_input(&format!("{prefix}_ffn2_w"), &[dim, ffn_dim]);
    let ffn1 = b.add_linear(normed2, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);

    // Residual 2
    b.add_binary_add(res1, ffn2, &shape)
}

/// Build bindings for a single encoder block.
fn encoder_block_bindings(dim: usize, ffn_dim: usize) -> Vec<TensorParamBinding> {
    vec![
        // ln1: weight, bias, eps
        ones(&[dim]),
        bias_zero(&[dim]),
        eps_binding(),
        // MHA: Q, K, V, O weights
        weight(&[dim, dim]),
        weight(&[dim, dim]),
        weight(&[dim, dim]),
        weight(&[dim, dim]),
        // ln2: weight, bias, eps
        ones(&[dim]),
        bias_zero(&[dim]),
        eps_binding(),
        // FFN: ffn1_w, ffn2_w
        weight(&[ffn_dim, dim]),
        weight(&[dim, ffn_dim]),
    ]
}

// ===========================================================================
// 1. Mel spectrogram feature extraction bounds (IBP)
// ===========================================================================

#[test]
fn test_audio_mel_spectrogram_bounds_ibp() {
    // Mel spectrogram: audio -> Linear (STFT approx) -> power -> mel projection
    let mut b = TensorBlockBuilder::new("audio_mel_spectrogram");
    let input = b.add_input("audio", &[AUDIO_LEN]);

    // STFT approximation: Linear over the 1D audio vector yields a 1D
    // [FRAMES*MEL_BINS*2] vector (a Linear replaces the last dim with
    // out_features; it does not reshape), then reshape to [FRAMES, MEL_BINS*2].
    // The reshape is element-count preserving (FRAMES*MEL_BINS*2 elements).
    let stft_w = b.add_input("stft_w", &[FRAMES * MEL_BINS * 2, AUDIO_LEN]);
    let stft_flat = b.add_linear(input, stft_w, None, &[FRAMES * MEL_BINS * 2]);
    let stft = b.add_reshape(stft_flat, &[FRAMES, MEL_BINS * 2]);

    // Split real/imag and compute power = real^2 + imag^2 (approximated via mul)
    let real = b.add_narrow(stft, 1, 0, MEL_BINS, &[FRAMES, MEL_BINS]);
    let imag = b.add_narrow(stft, 1, MEL_BINS, MEL_BINS, &[FRAMES, MEL_BINS]);
    let real_sq = b.add_binary_mul(real, real, &[FRAMES, MEL_BINS]);
    let imag_sq = b.add_binary_mul(imag, imag, &[FRAMES, MEL_BINS]);
    let power = b.add_binary_add(real_sq, imag_sq, &[FRAMES, MEL_BINS]);

    // Mel filterbank: Linear [FRAMES, MEL_BINS] -> [FRAMES, MEL_BINS]
    let mel_w = b.add_input("mel_w", &[MEL_BINS, MEL_BINS]);
    let out = b.add_linear(power, mel_w, None, &[FRAMES, MEL_BINS]);
    let def = b.build(out).expect("valid mel spectrogram kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FRAMES * MEL_BINS * 2, AUDIO_LEN]),
        weight(&[MEL_BINS, MEL_BINS]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = audio_bounds(AUDIO_LEN);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[FRAMES, MEL_BINS]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio mel spectrogram IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. VAD speech/silence probability bounds [0,1] (IBP)
// ===========================================================================

#[test]
fn test_audio_vad_probability_bounds_ibp() {
    // Silero VAD: Conv1d features -> Linear -> sigmoid -> [0, 1]
    let vad_hidden = 16;
    let mut b = TensorBlockBuilder::new("audio_vad_probability");
    let input = b.add_input("features", &[FRAMES, vad_hidden]);

    // Linear: [FRAMES, vad_hidden] -> [FRAMES, 1]
    let w = b.add_input("vad_w", &[1, vad_hidden]);
    let bias = b.add_input("vad_b", &[1]);
    let logit = b.add_linear(input, w, Some(bias), &[FRAMES, 1]);

    // Sigmoid: output in [0, 1]
    let out = b.add_sigmoid(logit, &[FRAMES, 1]);
    let def = b.build(out).expect("valid VAD kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[1, vad_hidden]),
        bias_zero(&[1]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(FRAMES, vad_hidden, 2.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio VAD probability IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid outputs must be in [0, 1]
    assert!(
        lo_min >= -1e-5,
        "sigmoid lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "sigmoid upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 3. Whisper encoder self-attention bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_audio_whisper_encoder_self_attention_ibp_crown() {
    let mut b = TensorBlockBuilder::new("audio_whisper_enc_self_attn");
    let input = b.add_input("x", &[FRAMES, D_MODEL]);
    let out = add_encoder_block(&mut b, input, FRAMES, D_MODEL, FFN_DIM, NUM_HEADS, "enc0");
    let def = b.build(out).expect("valid encoder block kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(D_MODEL, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(FRAMES, D_MODEL, 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Audio Whisper encoder self-attn IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN should also produce valid bounds
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Audio Whisper encoder self-attn CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 4. Whisper decoder cross-attention bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_audio_whisper_decoder_cross_attention_ibp_crown() {
    // Cross-attention: decoder queries attend to encoder memory
    let mut b = TensorBlockBuilder::new("audio_whisper_dec_cross_attn");
    let q_input = b.add_input("decoder_q", &[MAX_DECODE_LEN, D_MODEL]);
    let kv_input = b.add_input("encoder_kv", &[FRAMES, D_MODEL]);

    let qw = b.add_input("cross_q_w", &[D_MODEL, D_MODEL]);
    let kw = b.add_input("cross_k_w", &[D_MODEL, D_MODEL]);
    let vw = b.add_input("cross_v_w", &[D_MODEL, D_MODEL]);
    let ow = b.add_input("cross_o_w", &[D_MODEL, D_MODEL]);
    let cross_attn = b
        .add_multi_head_cross_attention(
            q_input,
            kv_input,
            qw,
            kw,
            vw,
            ow,
            NUM_HEADS,
            AttentionMask::Standard,
            &[MAX_DECODE_LEN, D_MODEL],
        )
        .expect("valid cross-attention");
    let def = b.build(cross_attn).expect("valid cross-attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // decoder_q
        TensorParamBinding::ConstantTensor(
            // encoder_kv (fixed encoder output)
            ArrayD::from_elem(IxDyn(&[FRAMES, D_MODEL]), WEIGHT_MAG),
        ),
        weight(&[D_MODEL, D_MODEL]),
        weight(&[D_MODEL, D_MODEL]),
        weight(&[D_MODEL, D_MODEL]),
        weight(&[D_MODEL, D_MODEL]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(MAX_DECODE_LEN, D_MODEL, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Audio Whisper cross-attn IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Audio Whisper cross-attn CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 5. Log-mel feature normalization bounds (IBP)
// ===========================================================================

#[test]
fn test_audio_log_mel_normalization_ibp() {
    // LayerNorm on mel features: [FRAMES, MEL_BINS] -> [FRAMES, MEL_BINS]
    let mut b = TensorBlockBuilder::new("audio_log_mel_norm");
    let input = b.add_input("mel_features", &[FRAMES, MEL_BINS]);
    let ln_w = b.add_input("ln_w", &[MEL_BINS]);
    let ln_b = b.add_input("ln_b", &[MEL_BINS]);
    let eps = b.add_input("eps", &[1]);
    let out = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[FRAMES, MEL_BINS]);
    let def = b.build(out).expect("valid log-mel norm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[MEL_BINS]),
        bias_zero(&[MEL_BINS]),
        eps_binding(),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(FRAMES, MEL_BINS, 5.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[FRAMES, MEL_BINS]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio log-mel normalization IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. Token probability output bounds (softmax) (IBP)
// ===========================================================================

#[test]
fn test_audio_token_probability_softmax_ibp() {
    // LM head: Linear -> softmax -> token probabilities in [0, 1]
    let mut b = TensorBlockBuilder::new("audio_token_prob_softmax");
    let input = b.add_input("decoder_hidden", &[MAX_DECODE_LEN, D_MODEL]);
    let lm_w = b.add_input("lm_head_w", &[VOCAB_SIZE, D_MODEL]);
    let logits = b.add_linear(input, lm_w, None, &[MAX_DECODE_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[MAX_DECODE_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid token probability kernel");

    let bindings = vec![TensorParamBinding::Variable, weight(&[VOCAB_SIZE, D_MODEL])];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(MAX_DECODE_LEN, D_MODEL, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio token probability softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax outputs must be in [0, 1]
    assert!(
        lo_min >= -1e-5,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 7. Full VAD-to-Whisper pipeline composition (IBP)
// ===========================================================================

#[test]
fn test_audio_vad_to_whisper_pipeline_ibp() {
    // VAD features -> Linear -> sigmoid (VAD decision)
    // Then mel features -> encoder block -> LN -> output
    // Both share the same audio-domain input bounds.
    let mut b = TensorBlockBuilder::new("audio_vad_to_whisper_pipeline");
    let mel_input = b.add_input("mel_features", &[FRAMES, MEL_BINS]);

    // Feature projection: [FRAMES, MEL_BINS] -> [FRAMES, D_MODEL]
    let proj_w = b.add_input("feat_proj_w", &[D_MODEL, MEL_BINS]);
    let proj_b = b.add_input("feat_proj_b", &[D_MODEL]);
    let projected = b.add_linear(mel_input, proj_w, Some(proj_b), &[FRAMES, D_MODEL]);

    // One encoder block
    let enc_out = add_encoder_block(
        &mut b, projected, FRAMES, D_MODEL, FFN_DIM, NUM_HEADS, "enc0",
    );

    // Final LayerNorm
    let ln_w = b.add_input("final_ln_w", &[D_MODEL]);
    let ln_b = b.add_input("final_ln_b", &[D_MODEL]);
    let eps = b.add_input("final_eps", &[1]);
    let out = b.add_layer_norm(enc_out, eps, 1, ln_w, ln_b, &[FRAMES, D_MODEL]);
    let def = b.build(out).expect("valid VAD-to-Whisper pipeline kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable, // mel_features
        weight(&[D_MODEL, MEL_BINS]),
        bias_zero(&[D_MODEL]),
    ];
    bindings.extend(encoder_block_bindings(D_MODEL, FFN_DIM));
    bindings.push(ones(&[D_MODEL])); // final LN weight
    bindings.push(bias_zero(&[D_MODEL])); // final LN bias
    bindings.push(eps_binding()); // final LN eps

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(FRAMES, MEL_BINS, 2.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[FRAMES, D_MODEL]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio VAD-to-Whisper pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 8. Audio preprocessing (resampling) bounds (IBP)
// ===========================================================================

#[test]
fn test_audio_preprocessing_resampling_ibp() {
    // Resampling approximated as Conv1d: [1, AUDIO_LEN] -> [1, AUDIO_LEN/2]
    let in_ch = 1;
    let out_len = AUDIO_LEN / 2;
    let kernel_size = 4;

    let mut b = TensorBlockBuilder::new("audio_resample");
    let input = b.add_input("audio", &[in_ch, AUDIO_LEN]);
    let conv_w = b.add_input("resample_w", &[in_ch, in_ch, kernel_size]);
    let conv_b = b.add_input("resample_b", &[in_ch]);
    let out = b.add_conv1d(input, conv_w, Some(conv_b), 2, 1, &[in_ch, out_len]);
    let def = b.build(out).expect("valid resampling kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[in_ch, in_ch, kernel_size]),
        bias_zero(&[in_ch]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[in_ch, AUDIO_LEN]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[in_ch, AUDIO_LEN]), 1.0f32),
    )
    .expect("valid audio bounds");

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[in_ch, out_len]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio resampling IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 9. Whisper encoder layer norm bounds (IBP)
// ===========================================================================

#[test]
fn test_audio_whisper_encoder_layernorm_ibp() {
    // Verify LayerNorm stabilizes bounds on wide encoder features
    let mut b = TensorBlockBuilder::new("audio_whisper_enc_ln");
    let input = b.add_input("enc_features", &[FRAMES, D_MODEL]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let out = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[FRAMES, D_MODEL]);
    let def = b.build(out).expect("valid encoder LN kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[D_MODEL]),
        bias_zero(&[D_MODEL]),
        eps_binding(),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(FRAMES, D_MODEL, 10.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio Whisper encoder LN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. Whisper decoder causal mask bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_audio_whisper_decoder_causal_mask_ibp_crown() {
    // Decoder with causal mask: ensures autoregressive attention bounds
    let mut b = TensorBlockBuilder::new("audio_whisper_dec_causal");
    let input = b.add_input("dec_tokens", &[MAX_DECODE_LEN, D_MODEL]);

    // Pre-norm
    let ln_w = b.add_input("dec_ln1_w", &[D_MODEL]);
    let ln_b = b.add_input("dec_ln1_b", &[D_MODEL]);
    let eps = b.add_input("dec_ln1_eps", &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[MAX_DECODE_LEN, D_MODEL]);

    // Causal self-attention
    let qw = b.add_input("dec_q_w", &[D_MODEL, D_MODEL]);
    let kw = b.add_input("dec_k_w", &[D_MODEL, D_MODEL]);
    let vw = b.add_input("dec_v_w", &[D_MODEL, D_MODEL]);
    let ow = b.add_input("dec_o_w", &[D_MODEL, D_MODEL]);
    let attn = b
        .add_multi_head_attention(
            normed,
            qw,
            kw,
            vw,
            ow,
            NUM_HEADS,
            AttentionMask::Causal,
            &[MAX_DECODE_LEN, D_MODEL],
        )
        .expect("valid causal MHA");

    // Residual
    let out = b.add_binary_add(input, attn, &[MAX_DECODE_LEN, D_MODEL]);
    let def = b.build(out).expect("valid decoder causal kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[D_MODEL]),
        bias_zero(&[D_MODEL]),
        eps_binding(),
        weight(&[D_MODEL, D_MODEL]),
        weight(&[D_MODEL, D_MODEL]),
        weight(&[D_MODEL, D_MODEL]),
        weight(&[D_MODEL, D_MODEL]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(MAX_DECODE_LEN, D_MODEL, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Audio Whisper decoder causal IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Audio Whisper decoder causal CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 11. Positional encoding for audio frames (IBP)
// ===========================================================================

#[test]
fn test_audio_positional_encoding_ibp() {
    // Sinusoidal PE added to mel embeddings: [FRAMES, D_MODEL] + PE[FRAMES, D_MODEL]
    let mut b = TensorBlockBuilder::new("audio_pos_encoding");
    let input = b.add_input("mel_embeddings", &[FRAMES, D_MODEL]);
    let pe = b.add_input("pos_encoding", &[FRAMES, D_MODEL]);
    let out = b.add_binary_add(input, pe, &[FRAMES, D_MODEL]);
    let def = b.build(out).expect("valid pos encoding kernel");

    // Sinusoidal PE is bounded in [-1, 1] per element
    let pe_data = ArrayD::from_elem(IxDyn(&[FRAMES, D_MODEL]), 0.5f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(FRAMES, D_MODEL, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio positional encoding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // PE addition shifts bounds but should remain finite
    assert!(lo_min < 0.0, "lower bound should be negative");
    assert!(hi_max > 0.0, "upper bound should be positive");
}

// ===========================================================================
// 12. Multi-head attention output bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_audio_multi_head_attention_output_ibp_crown() {
    // Standalone MHA: [FRAMES, D_MODEL] -> MHA -> [FRAMES, D_MODEL]
    let mut b = TensorBlockBuilder::new("audio_mha_output");
    let input = b.add_input("x", &[FRAMES, D_MODEL]);
    let qw = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let kw = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let vw = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let ow = b.add_input("o_w", &[D_MODEL, D_MODEL]);
    let out = b
        .add_multi_head_attention(
            input,
            qw,
            kw,
            vw,
            ow,
            NUM_HEADS,
            AttentionMask::Standard,
            &[FRAMES, D_MODEL],
        )
        .expect("valid MHA");
    let def = b.build(out).expect("valid MHA kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[D_MODEL, D_MODEL]),
        weight(&[D_MODEL, D_MODEL]),
        weight(&[D_MODEL, D_MODEL]),
        weight(&[D_MODEL, D_MODEL]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(FRAMES, D_MODEL, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Audio MHA output IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Audio MHA output CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 13. Feed-forward network intermediate bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_audio_ffn_intermediate_bounds_ibp_crown() {
    // FFN: Linear(D_MODEL, FFN_DIM) -> GELU -> Linear(FFN_DIM, D_MODEL)
    let mut b = TensorBlockBuilder::new("audio_ffn_intermediate");
    let input = b.add_input("x", &[FRAMES, D_MODEL]);

    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn1_b = b.add_input("ffn1_b", &[FFN_DIM]);
    let ffn1 = b.add_linear(input, ffn1_w, Some(ffn1_b), &[FRAMES, FFN_DIM]);
    let act = b.add_gelu(ffn1, &[FRAMES, FFN_DIM]);

    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);
    let ffn2_b = b.add_input("ffn2_b", &[D_MODEL]);
    let out = b.add_linear(act, ffn2_w, Some(ffn2_b), &[FRAMES, D_MODEL]);
    let def = b.build(out).expect("valid FFN kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, D_MODEL]),
        bias_zero(&[FFN_DIM]),
        weight(&[D_MODEL, FFN_DIM]),
        bias_zero(&[D_MODEL]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(FRAMES, D_MODEL, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Audio FFN intermediate IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Audio FFN intermediate CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 14. KV cache bounds in decoder (IBP)
// ===========================================================================

#[test]
fn test_audio_kv_cache_bounds_ibp() {
    // KV cache: concatenate cached K/V with new K/V, run attention.
    // Approximated as: [CACHE_LEN + 1, D_MODEL] input -> attention -> [1, D_MODEL] output
    let cache_len = 4;
    let full_kv_len = cache_len + 1;

    let mut b = TensorBlockBuilder::new("audio_kv_cache");
    // Query: single new token [1, D_MODEL]
    let q_input = b.add_input("q_new", &[1, D_MODEL]);
    // KV: cached + new [full_kv_len, D_MODEL]
    let kv_input = b.add_input("kv_cached", &[full_kv_len, D_MODEL]);

    let qw = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let kw = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let vw = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let ow = b.add_input("o_w", &[D_MODEL, D_MODEL]);
    let out = b
        .add_multi_head_cross_attention(
            q_input,
            kv_input,
            qw,
            kw,
            vw,
            ow,
            NUM_HEADS,
            AttentionMask::Standard,
            &[1, D_MODEL],
        )
        .expect("valid KV cache attention");
    let def = b.build(out).expect("valid KV cache kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // q_new
        TensorParamBinding::ConstantTensor(
            // kv_cached (fixed past context)
            ArrayD::from_elem(IxDyn(&[full_kv_len, D_MODEL]), WEIGHT_MAG),
        ),
        weight(&[D_MODEL, D_MODEL]),
        weight(&[D_MODEL, D_MODEL]),
        weight(&[D_MODEL, D_MODEL]),
        weight(&[D_MODEL, D_MODEL]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(1, D_MODEL, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, D_MODEL]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio KV cache IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 15. Beam search score bounds (IBP)
// ===========================================================================

#[test]
fn test_audio_beam_search_score_bounds_ibp() {
    // Beam search: log_softmax on logits produces log-probabilities in (-inf, 0]
    // We use softmax + narrow to test the bounded portion.
    let beam_width = 4;

    let mut b = TensorBlockBuilder::new("audio_beam_search_scores");
    let input = b.add_input("logits", &[1, VOCAB_SIZE]);

    // Log-softmax: output in (-inf, 0]
    let log_probs = b.add_log_softmax(input, -1, &[1, VOCAB_SIZE]);

    // Select top-k (approximated by narrow to first beam_width entries)
    let out = b.add_narrow(log_probs, 1, 0, beam_width, &[1, beam_width]);
    let def = b.build(out).expect("valid beam search kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(1, VOCAB_SIZE, 5.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio beam search scores IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Log-softmax upper bound should be <= 0 (with tolerance)
    assert!(
        hi_max <= 0.0 + 1e-5,
        "log_softmax upper bound must be <= 0, got {hi_max}"
    );
}

// ===========================================================================
// 16. Audio energy / volume bounds (IBP)
// ===========================================================================

#[test]
fn test_audio_energy_volume_bounds_ibp() {
    // Audio energy: x^2 averaged over a window. Power is always >= 0.
    // Approximated as: x * x -> reduce_mean
    let window_size = 16;

    let mut b = TensorBlockBuilder::new("audio_energy");
    let input = b.add_input("audio_window", &[window_size]);

    // Square: element-wise x * x
    let squared = b.add_binary_mul(input, input, &[window_size]);

    // Mean reduction: [window_size] -> [1] (approximated by linear with 1/N weights)
    let mean_w = b.add_input("mean_w", &[1, window_size]);
    let out = b.add_linear(squared, mean_w, None, &[1]);
    let def = b.build(out).expect("valid energy kernel");

    let avg_weight = 1.0 / window_size as f32;
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, window_size]), avg_weight)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[window_size], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio energy IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Energy (mean of squares averaged with positive weights) has finite bounds
    assert!(lo_min.is_finite(), "energy lower bound must be finite");
    assert!(hi_max.is_finite(), "energy upper bound must be finite");
}

// ===========================================================================
// 17. Speech segment concatenation bounds (IBP)
// ===========================================================================

#[test]
fn test_audio_speech_segment_concatenation_ibp() {
    // Concatenating two speech segments and projecting to a common space.
    // Segment A: [FRAMES, D_MODEL], Segment B: [FRAMES, D_MODEL]
    // Concat: [2*FRAMES, D_MODEL] -> Linear -> [2*FRAMES, D_MODEL]
    let concat_len = 2 * FRAMES;

    let mut b = TensorBlockBuilder::new("audio_segment_concat");
    let seg_a = b.add_input("segment_a", &[FRAMES, D_MODEL]);
    let seg_b = b.add_input("segment_b", &[FRAMES, D_MODEL]);

    // Concatenate along sequence dimension
    let concat = b.add_concat(&[seg_a, seg_b], 0, &[concat_len, D_MODEL]);

    // Project concatenated features
    let proj_w = b.add_input("proj_w", &[D_MODEL, D_MODEL]);
    let proj_b = b.add_input("proj_b", &[D_MODEL]);
    let out = b.add_linear(concat, proj_w, Some(proj_b), &[concat_len, D_MODEL]);
    let def = b.build(out).expect("valid segment concat kernel");

    let seg_b_data = ArrayD::from_elem(IxDyn(&[FRAMES, D_MODEL]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,                   // segment_a
        TensorParamBinding::ConstantTensor(seg_b_data), // segment_b (fixed)
        weight(&[D_MODEL, D_MODEL]),
        bias_zero(&[D_MODEL]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(FRAMES, D_MODEL, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[concat_len, D_MODEL]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio segment concatenation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 18. CTC decoder output bounds (IBP)
// ===========================================================================

#[test]
fn test_audio_ctc_decoder_output_ibp() {
    // CTC decoder: encoder features -> Linear -> log_softmax -> CTC output
    // VOCAB_SIZE + 1 for blank token
    let ctc_vocab = VOCAB_SIZE + 1;

    let mut b = TensorBlockBuilder::new("audio_ctc_decoder");
    let input = b.add_input("encoder_out", &[FRAMES, D_MODEL]);

    // CTC projection: [FRAMES, D_MODEL] -> [FRAMES, ctc_vocab]
    let ctc_w = b.add_input("ctc_w", &[ctc_vocab, D_MODEL]);
    let ctc_b = b.add_input("ctc_b", &[ctc_vocab]);
    let logits = b.add_linear(input, ctc_w, Some(ctc_b), &[FRAMES, ctc_vocab]);

    // Log-softmax over vocabulary dimension
    let out = b.add_log_softmax(logits, -1, &[FRAMES, ctc_vocab]);
    let def = b.build(out).expect("valid CTC decoder kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[ctc_vocab, D_MODEL]),
        bias_zero(&[ctc_vocab]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(FRAMES, D_MODEL, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[FRAMES, ctc_vocab]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Audio CTC decoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Log-softmax upper bound should be <= 0
    assert!(
        hi_max <= 0.0 + 1e-5,
        "CTC log_softmax upper bound must be <= 0, got {hi_max}"
    );
    assert!(lo_min.is_finite(), "CTC lower bound must be finite");
}
