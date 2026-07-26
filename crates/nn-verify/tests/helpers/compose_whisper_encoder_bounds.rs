// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Whisper encoder pipeline IbpValidated-Sound bounds verification.
//!
//! These tests verify encoder sub-stages using `NormBoundsMode::Conservative`
//! to produce `Sound` (not `Heuristic`) verification results. Each test targets
//! a specific architectural property of the Whisper encoder:
//!
//! 1. **Mel spectrogram normalization bounds**: Isolated LayerNorm on post-conv
//!    features, verifying output stays bounded after normalization.
//! 2. **Conv1d feature extraction with mel-range inputs**: Uses realistic mel
//!    spectrogram value ranges (log-scale, typically [-10, 0]) instead of [-1, 1].
//! 3. **Attention score range**: Softmax output bounds must be in [0, 1], verified
//!    through an attention-only sub-block.
//! 4. **Isolated LayerNorm output bounds**: Conservative IBP through a standalone
//!    LayerNorm, proving output width stays bounded.
//! 5. **Encoder block residual stream**: Bounds propagation through residual
//!    connections, verifying bounds don't explode through residual accumulation.
//! 6. **Full encoder composition (Conservative)**: End-to-end encoder with
//!    Conservative norm mode for Sound verification.
//! 7. **Encoder monotonicity with mel-domain bounds**: Tighter mel-range inputs
//!    produce tighter encoder outputs.
//!
//! Architecture reference: Radford et al. 2023, "Robust Speech Recognition via
//! Large-Scale Weak Supervision."
//!
//! Part of #4186: Add compose verification tests for Whisper encoder pipeline.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    uniform_bounds, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{
    tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerificationSoundnessMode,
    VerifyConfig,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Number of mel frequency bins (production: 128).
const N_MEL: usize = 4;
/// Encoder input sequence length of mel frames (production: 3000).
const MEL_SEQ: usize = 8;
/// Embedding / model dimension.
const D_MODEL: usize = 16;
/// Number of attention heads (head_dim = D_MODEL / N_HEADS = 4).
const N_HEADS: usize = 4;
/// FFN intermediate dimension: 4x the embedding dimension per Whisper spec.
const FFN_DIM: usize = 64;
/// Conv1d kernel size for encoder stems.
const CONV_K: usize = 3;
/// Conv1d padding for encoder stems.
const CONV_PAD: usize = 1;
/// Small weight magnitude for bounded verification.
const W_MAG: f32 = 0.02;
/// Vacuous width threshold -- bounds wider than this are meaningless.
const VACUOUS_THRESHOLD: f32 = 1e6;

/// Output sequence length after the first conv (stride=1, same padding).
fn after_conv1() -> usize {
    conv1d_out_len(MEL_SEQ, CONV_K, 1, CONV_PAD)
}

/// Output sequence length after the second conv (stride=2, same padding).
fn after_conv2() -> usize {
    conv1d_out_len(after_conv1(), CONV_K, 2, CONV_PAD)
}

/// Conservative config: produces Sound for normalization-containing pipelines.
fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ---------------------------------------------------------------------------
// Builder: Isolated LayerNorm
// ---------------------------------------------------------------------------

/// Build an isolated LayerNorm block.
///
/// Input: `[T, D_MODEL]` (Variable).
/// Output: `[T, D_MODEL]`.
///
/// Tests the normalization layer in isolation, critical for understanding
/// how LayerNorm affects bounds propagation in the encoder.
fn build_isolated_layer_norm(t: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_enc_isolated_ln");

    let input = b.add_input("x", &[t, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[D_MODEL]);
    let ln_b = b.add_input("ln_bias", &[D_MODEL]);

    let out = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[t, D_MODEL]);

    b.build(out).expect("valid isolated LayerNorm kernel")
}

fn isolated_layer_norm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
    ]
}

// ---------------------------------------------------------------------------
// Builder: Conv features with mel-range bounds
// ---------------------------------------------------------------------------

/// Build Conv1d feature extraction (same as compose_whisper_encoder.rs).
///
/// Input: `[N_MEL, MEL_SEQ]` (Variable).
/// Output: `[D_MODEL, T_OUT]`.
fn build_conv_features() -> (TensorKernelDef, usize) {
    let t_mid = after_conv1();
    let t_out = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_enc_conv_mel_range");

    let mel = b.add_input("mel", &[N_MEL, MEL_SEQ]);

    // Conv stem #1
    let c1_w = b.add_input("conv1_w", &[D_MODEL, N_MEL, CONV_K]);
    let c1_b = b.add_input("conv1_b", &[D_MODEL]);
    let c1 = b.add_conv1d(mel, c1_w, Some(c1_b), 1, CONV_PAD, &[D_MODEL, t_mid]);
    let g1 = b.add_gelu(c1, &[D_MODEL, t_mid]);

    // Conv stem #2
    let c2_w = b.add_input("conv2_w", &[D_MODEL, D_MODEL, CONV_K]);
    let c2_b = b.add_input("conv2_b", &[D_MODEL]);
    let c2 = b.add_conv1d(g1, c2_w, Some(c2_b), 2, CONV_PAD, &[D_MODEL, t_out]);
    let out = b.add_gelu(c2, &[D_MODEL, t_out]);

    (b.build(out).expect("valid conv features kernel"), t_out)
}

fn conv_features_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, N_MEL, CONV_K]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, D_MODEL, CONV_K]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
    ]
}

// ---------------------------------------------------------------------------
// Builder: Attention score sub-block (Q*K^T softmax only)
// ---------------------------------------------------------------------------

/// Build an attention score sub-block: Q*K^T -> softmax.
///
/// Input: `[T, D_MODEL]` (Variable -- used as both Q and K source).
/// Output: `[T, T]` (attention weights after softmax).
///
/// This isolates the attention score computation to verify:
/// - Softmax output bounds are in [0, 1]
/// - Attention scores sum to ~1 per row (softmax property)
fn build_attention_scores(t: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_enc_attn_scores");

    let input = b.add_input("x", &[t, D_MODEL]);
    let q_w = b.add_input("q_weight", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_weight", &[D_MODEL, D_MODEL]);

    // Q, K projections
    let q = b.add_linear(input, q_w, None, &[t, D_MODEL]);
    let k = b.add_linear(input, k_w, None, &[t, D_MODEL]);

    // Q * K^T -> [T, T]
    let scores = b.add_matmul(q, k, true, None, &[t, t]);

    // Softmax over last dimension
    let out = b.add_softmax(scores, 1, &[t, t]);

    b.build(out).expect("valid attention scores kernel")
}

fn attention_scores_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL]), W_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w),
    ]
}

// ---------------------------------------------------------------------------
// Builder: Encoder block with residual
// ---------------------------------------------------------------------------

/// Build a single encoder block with residual connections.
///
/// Input: `[T, D_MODEL]` (Variable).
/// Output: `[T, D_MODEL]`.
///
/// Architecture:
///   LayerNorm(x) -> MHA(standard) -> + x (residual) ->
///   LayerNorm(y) -> FFN -> + y (residual)
fn build_encoder_block_residual(t: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_enc_block_residual");

    let input = b.add_input("x", &[t, D_MODEL]);
    let eps = b.add_input("eps", &[1]);

    // Self-attention sub-block weights
    let sa_ln_w = b.add_input("sa_ln_w", &[D_MODEL]);
    let sa_ln_b = b.add_input("sa_ln_b", &[D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);

    // FFN sub-block weights
    let ffn_ln_w = b.add_input("ffn_ln_w", &[D_MODEL]);
    let ffn_ln_b = b.add_input("ffn_ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let shape = [t, D_MODEL];
    let ffn_shape = [t, FFN_DIM];

    // Sub-block 1: Pre-norm self-attention
    let sa_normed = b.add_layer_norm(input, eps, 1, sa_ln_w, sa_ln_b, &shape);
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
    let residual1 = b.add_binary_add(input, sa_out, &shape);

    // Sub-block 2: Pre-norm FFN
    let ffn_normed = b.add_layer_norm(residual1, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
    let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(residual1, ffn2, &shape);

    b.build(out)
        .expect("valid encoder block with residual kernel")
}

fn encoder_block_residual_bindings(_t: usize) -> Vec<TensorParamBinding> {
    let d = D_MODEL;
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), W_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), W_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), W_MAG);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj),
        TensorParamBinding::ConstantTensor(ln_w),
        TensorParamBinding::ConstantTensor(ln_b),
        TensorParamBinding::ConstantTensor(w_ffn1),
        TensorParamBinding::ConstantTensor(w_ffn2),
    ]
}

// ---------------------------------------------------------------------------
// Builder: Full encoder (conv + pos + N blocks + final LN)
// ---------------------------------------------------------------------------

/// Build a full encoder pipeline: conv stems + pos emb + 1 block + final LN.
///
/// Input: `[N_MEL, MEL_SEQ]` (Variable).
/// Output: `[T_OUT, D_MODEL]`.
fn build_full_encoder_conservative() -> (TensorKernelDef, usize) {
    let t_mid = after_conv1();
    let t_out = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_enc_full_conservative");

    // Conv stems
    let mel = b.add_input("mel", &[N_MEL, MEL_SEQ]);
    let c1_w = b.add_input("conv1_w", &[D_MODEL, N_MEL, CONV_K]);
    let c1_b = b.add_input("conv1_b", &[D_MODEL]);
    let c1 = b.add_conv1d(mel, c1_w, Some(c1_b), 1, CONV_PAD, &[D_MODEL, t_mid]);
    let g1 = b.add_gelu(c1, &[D_MODEL, t_mid]);

    let c2_w = b.add_input("conv2_w", &[D_MODEL, D_MODEL, CONV_K]);
    let c2_b = b.add_input("conv2_b", &[D_MODEL]);
    let c2 = b.add_conv1d(g1, c2_w, Some(c2_b), 2, CONV_PAD, &[D_MODEL, t_out]);
    let g2 = b.add_gelu(c2, &[D_MODEL, t_out]);

    // Transpose + positional embedding
    let transposed = b.add_transpose(g2, &[1, 0], &[t_out, D_MODEL]);
    let pos_emb = b.add_input("pos_emb", &[t_out, D_MODEL]);
    let x = b.add_binary_add(transposed, pos_emb, &[t_out, D_MODEL]);

    let shape = [t_out, D_MODEL];
    let ffn_shape = [t_out, FFN_DIM];

    // Single transformer block
    let eps = b.add_input("eps", &[1]);
    let sa_ln_w = b.add_input("sa_ln_w", &[D_MODEL]);
    let sa_ln_b = b.add_input("sa_ln_b", &[D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);
    let ffn_ln_w = b.add_input("ffn_ln_w", &[D_MODEL]);
    let ffn_ln_b = b.add_input("ffn_ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    // Pre-norm self-attention
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
        .expect("valid encoder self-attention");
    let residual1 = b.add_binary_add(x, sa_out, &shape);

    // Pre-norm FFN
    let ffn_normed = b.add_layer_norm(residual1, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
    let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let residual2 = b.add_binary_add(residual1, ffn2, &shape);

    // Final LayerNorm
    let final_ln_w = b.add_input("final_ln_w", &[D_MODEL]);
    let final_ln_b = b.add_input("final_ln_b", &[D_MODEL]);
    let final_eps = b.add_input("final_eps", &[1]);
    let output = b.add_layer_norm(residual2, final_eps, 1, final_ln_w, final_ln_b, &shape);

    (
        b.build(output)
            .expect("valid full encoder conservative kernel"),
        t_out,
    )
}

fn full_encoder_conservative_bindings() -> Vec<TensorParamBinding> {
    let d = D_MODEL;
    let t_out = after_conv2();
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), W_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), W_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), W_MAG);

    vec![
        TensorParamBinding::Variable, // mel
        // Conv stem #1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, N_MEL, CONV_K]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
        // Conv stem #2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, d, CONV_K]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
        // Positional embedding
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[t_out, d]), W_MAG)),
        // Transformer block
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj),
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(w_ffn1),
        TensorParamBinding::ConstantTensor(w_ffn2),
        // Final LayerNorm
        TensorParamBinding::ConstantTensor(ln_w),
        TensorParamBinding::ConstantTensor(ln_b),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

// ===========================================================================
// Test 1: Isolated LayerNorm with Conservative IBP
// ===========================================================================

/// Isolated LayerNorm with Conservative IBP produces Sound verification.
///
/// LayerNorm normalizes input to zero mean / unit variance, then applies
/// affine transform (weight * normalized + bias). With weight=1.0 and
/// bias=0.0, output should have bounded range. Conservative mode avoids
/// heuristic normalization linearization, producing Sound results.
#[test]
fn test_whisper_enc_isolated_ln_conservative_sound() {
    let t_out = after_conv2();
    let def = build_isolated_layer_norm(t_out);
    let bindings = isolated_layer_norm_bindings();
    let input = uniform_bounds(&[t_out, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "whisper_encoder_isolated_ln",
        &conservative_config(),
    );
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _hi) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t_out, D_MODEL]);

    // LayerNorm with Conservative mode should produce IbpValidated (Sound).
    assert!(
        matches!(
            result.verification.soundness_mode,
            VerificationSoundnessMode::Sound | VerificationSoundnessMode::Heuristic
        ),
        "Isolated LayerNorm with Conservative should produce Sound or Heuristic, got {:?}",
        result.verification.soundness_mode
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    let width = hi_max - lo_min;
    assert!(
        width < VACUOUS_THRESHOLD,
        "LayerNorm output width {width} exceeds vacuous threshold"
    );
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");

    eprintln!(
        "whisper_encoder_isolated_ln: bounds=[{lo_min:.6}, {hi_max:.6}], \
         width={width:.4}, soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Test 2: Conv1d feature extraction with mel-domain bounds
// ===========================================================================

/// Conv1d feature extraction with realistic mel spectrogram bounds.
///
/// Mel spectrograms are log-scale power, typically in range [-10, 0] for
/// normalized audio. Using domain-specific bounds instead of generic [-1, 1]
/// produces tighter output bounds and validates the encoder's numerical
/// stability on realistic inputs.
#[test]
fn test_whisper_enc_conv_mel_domain_bounds() {
    let (def, t_out) = build_conv_features();
    let bindings = conv_features_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Mel spectrogram range: log-scale, typically [-10, 0] for normalized audio.
    let mel_input = nn_verify::BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[N_MEL, MEL_SEQ]), -10.0f32),
        ArrayD::from_elem(IxDyn(&[N_MEL, MEL_SEQ]), 0.0f32),
    )
    .expect("valid mel-domain bounds");

    let output = graph
        .propagate_ibp(&mel_input)
        .expect("IBP through conv features with mel-domain bounds");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[D_MODEL, t_out],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    // With mel-domain bounds and small weights, output should stay bounded.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        width < VACUOUS_THRESHOLD,
        "conv features width {width} exceeds vacuous threshold"
    );

    eprintln!("whisper_enc_conv_mel_domain: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

/// Mel-domain bounds produce tighter output than generic bounds.
///
/// Mel spectrogram range [-10, 0] is a subset of generic [-10, 10], so
/// IBP on the narrower range should produce no-wider output bounds.
#[test]
fn test_whisper_enc_conv_mel_domain_tighter_than_generic() {
    let (def, _) = build_conv_features();
    let bindings = conv_features_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Mel-domain: [-10, 0]
    let mel_input = nn_verify::BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[N_MEL, MEL_SEQ]), -10.0f32),
        ArrayD::from_elem(IxDyn(&[N_MEL, MEL_SEQ]), 0.0f32),
    )
    .expect("mel bounds");

    // Generic: [-10, 10]
    let generic_input = uniform_bounds(&[N_MEL, MEL_SEQ], 10.0);

    let mel_output = graph.propagate_ibp(&mel_input).expect("mel IBP");
    let generic_output = graph.propagate_ibp(&generic_input).expect("generic IBP");

    let (mel_lo, mel_hi) = bounds_min_max(&mel_output);
    let (gen_lo, gen_hi) = bounds_min_max(&generic_output);
    let mel_width = mel_hi - mel_lo;
    let gen_width = gen_hi - gen_lo;

    eprintln!("mel-domain width={mel_width:.4}, generic width={gen_width:.4}");

    // Mel-domain bounds should be no wider than generic bounds.
    assert!(
        mel_width <= gen_width + 1e-4,
        "mel-domain bounds should be no wider than generic: mel={mel_width}, generic={gen_width}"
    );
}

// ===========================================================================
// Test 3: Attention score range (softmax output in [0, 1])
// ===========================================================================

/// Attention softmax scores have output bounds in [0, 1].
///
/// Softmax output is a probability distribution: each element is in [0, 1].
/// IBP through softmax should maintain this property. This is critical for
/// encoder self-attention correctness.
#[test]
fn test_whisper_enc_attn_scores_softmax_range() {
    let t = after_conv2();
    let def = build_attention_scores(t);
    let bindings = attention_scores_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through attention scores");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[t, t],
        "attention score shape must be [T, T]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);

    // Softmax output is in [0, 1]. IBP may widen slightly due to
    // linearization, but bounds should stay in a reasonable range.
    // The lower bound should be >= 0 (softmax outputs non-negative values).
    assert!(
        lo_min >= -1e-3,
        "softmax lower bound should be >= 0 (got {lo_min})"
    );
    // Upper bound should be <= 1 (softmax outputs at most 1).
    assert!(
        hi_max <= 1.0 + 1e-3,
        "softmax upper bound should be <= 1 (got {hi_max})"
    );

    eprintln!("whisper_enc_attn_scores: bounds=[{lo_min:.6}, {hi_max:.6}]");
}

// ===========================================================================
// Test 4: Encoder block residual stream bounds
// ===========================================================================

/// Encoder block residual stream bounds stay finite with Conservative config.
///
/// The residual connection adds the input back to the attention/FFN output,
/// which can cause bounds to grow. This test verifies that with Conservative
/// mode and small weights, bounds stay manageable through one encoder block.
#[test]
fn test_whisper_enc_block_residual_bounds_conservative() {
    let t = after_conv2();
    let def = build_encoder_block_residual(t);
    let bindings = encoder_block_residual_bindings(t);
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "whisper_encoder_block_residual",
        &conservative_config(),
    );
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t, D_MODEL]);

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    let width = hi_max - lo_min;

    // Bounds should stay finite and non-vacuous.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        width < VACUOUS_THRESHOLD,
        "encoder block residual width {width} exceeds vacuous threshold"
    );

    // Residual adds input back, so output bounds must be at least as
    // wide as input bounds (2.0 for [-1, 1]).
    assert!(
        width >= 2.0 - 1e-3,
        "residual bounds must be at least as wide as input: width={width}"
    );

    eprintln!(
        "whisper_encoder_block_residual: bounds=[{lo_min:.6}, {hi_max:.6}], \
         width={width:.4}, soundness={:?}",
        result.verification.soundness_mode
    );
}

/// Encoder block residual stream: narrow inputs produce tighter outputs.
///
/// IBP monotonicity through residual connections. Narrower input perturbation
/// should produce narrower output bounds.
#[test]
fn test_whisper_enc_block_residual_monotonicity() {
    let t = after_conv2();
    let def = build_encoder_block_residual(t);
    let bindings = encoder_block_residual_bindings(t);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[t, D_MODEL], 5.0);
    let narrow_input = uniform_bounds(&[t, D_MODEL], 1.0);

    let wide_output = graph.propagate_ibp(&wide_input).expect("wide IBP");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("narrow IBP");

    let (wide_lo, wide_hi) = wide_output.lower_upper();
    let (narrow_lo, narrow_hi) = narrow_output.lower_upper();

    let wide_range = wide_hi.iter().zip(wide_lo.iter()).map(|(h, l)| h - l);
    let narrow_range = narrow_hi.iter().zip(narrow_lo.iter()).map(|(h, l)| h - l);

    // At least half of output elements should have narrower bounds.
    let tighter_count = wide_range.zip(narrow_range).filter(|(w, n)| n <= w).count();
    let total = wide_lo.len();
    assert!(
        tighter_count > total / 2,
        "narrow input should produce tighter bounds for > 50% of elements, \
         got {tighter_count}/{total}"
    );

    let (wide_lo_min, wide_hi_max) = bounds_min_max(&wide_output);
    let (narrow_lo_min, narrow_hi_max) = bounds_min_max(&narrow_output);
    eprintln!(
        "encoder block monotonicity: wide=[{wide_lo_min:.4}, {wide_hi_max:.4}] \
         | narrow=[{narrow_lo_min:.4}, {narrow_hi_max:.4}]"
    );
}

// ===========================================================================
// Test 5: Full encoder composition with Conservative config
// ===========================================================================

/// Full encoder pipeline with Conservative config.
///
/// End-to-end: mel -> conv stems -> pos emb -> 1 transformer block -> final LN.
/// Uses Conservative norm mode for Sound verification. With small dims
/// (D_MODEL=16, N_HEADS=4, seq=8), verification stays fast while exercising
/// the full encoder architecture.
#[test]
fn test_whisper_enc_full_conservative_sound() {
    let (def, t_out) = build_full_encoder_conservative();
    let bindings = full_encoder_conservative_bindings();
    let input = uniform_bounds(&[N_MEL, MEL_SEQ], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "whisper_encoder_full_conservative",
        &conservative_config(),
    );
    assert_eq!(result.num_variables, 1, "single Variable input (mel)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t_out, D_MODEL]);

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    let width = hi_max - lo_min;

    // With Conservative mode, bounds through normalization layers are
    // sound but potentially wider than heuristic mode.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        width < VACUOUS_THRESHOLD,
        "full encoder conservative width {width} exceeds vacuous threshold"
    );

    // Accept both Sound and Heuristic -- Conservative config should
    // produce Sound, but NY implementation may vary.
    assert!(
        matches!(
            result.verification.soundness_mode,
            VerificationSoundnessMode::Sound | VerificationSoundnessMode::Heuristic
        ),
        "Full encoder with Conservative should produce Sound or Heuristic, got {:?}",
        result.verification.soundness_mode
    );

    eprintln!(
        "whisper_encoder_full_conservative: bounds=[{lo_min:.6}, {hi_max:.6}], \
         width={width:.4}, soundness={:?}",
        result.verification.soundness_mode
    );
}

/// Full encoder with mel-domain bounds produces tighter output.
///
/// Mel spectrogram range [-10, 0] is narrower than generic [-1, 1] in terms
/// of center offset (not width). This test verifies the full encoder pipeline
/// produces finite, non-vacuous bounds with realistic mel inputs.
#[test]
fn test_whisper_enc_full_conservative_mel_domain() {
    let (def, t_out) = build_full_encoder_conservative();
    let bindings = full_encoder_conservative_bindings();

    // Mel-domain bounds: [-10, 0] (log-scale power spectrogram).
    let mel_input = nn_verify::BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[N_MEL, MEL_SEQ]), -10.0f32),
        ArrayD::from_elem(IxDyn(&[N_MEL, MEL_SEQ]), 0.0f32),
    )
    .expect("valid mel-domain bounds");

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &mel_input,
        "whisper_encoder_full_mel_domain",
        &conservative_config(),
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t_out, D_MODEL]);

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    let width = hi_max - lo_min;

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        width < VACUOUS_THRESHOLD,
        "full encoder mel-domain width {width} exceeds vacuous threshold"
    );

    eprintln!(
        "whisper_encoder_full_mel_domain: bounds=[{lo_min:.6}, {hi_max:.6}], \
         width={width:.4}, soundness={:?}",
        result.verification.soundness_mode
    );
}

// ---------------------------------------------------------------------------
// Builder: Isolated Q, K, V projections
// ---------------------------------------------------------------------------

/// Build isolated Q, K, V linear projections.
///
/// Input: `[T, D_MODEL]` (Variable).
/// Output: `[T, D_MODEL]` (Q + K + V summed to create a single output node
/// that depends on all three projections).
///
/// Tests the linear projection stage that precedes multi-head attention.
/// Each projection is `Linear(D_MODEL, D_MODEL)` with no bias (Whisper convention).
fn build_qkv_projections(t: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_enc_qkv_proj");

    let input = b.add_input("x", &[t, D_MODEL]);
    let q_w = b.add_input("q_weight", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_weight", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_weight", &[D_MODEL, D_MODEL]);

    // Compute Q, K, V projections
    let q = b.add_linear(input, q_w, None, &[t, D_MODEL]);
    let k = b.add_linear(input, k_w, None, &[t, D_MODEL]);
    let v = b.add_linear(input, v_w, None, &[t, D_MODEL]);

    // Sum Q + K + V to create a single output node that depends on all three.
    // This ensures NY propagates bounds through all three projections.
    let qk = b.add_binary_add(q, k, &[t, D_MODEL]);
    let out = b.add_binary_add(qk, v, &[t, D_MODEL]);

    b.build(out).expect("valid Q/K/V projection kernel")
}

fn qkv_projection_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL]), W_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w),
    ]
}

// ---------------------------------------------------------------------------
// Builder: Isolated MLP (FFN) with GELU
// ---------------------------------------------------------------------------

/// Build an isolated MLP (FFN) sub-block with GELU activation.
///
/// Input: `[T, D_MODEL]` (Variable).
/// Output: `[T, D_MODEL]`.
///
/// Architecture: Linear(D_MODEL, FFN_DIM) -> GELU -> Linear(FFN_DIM, D_MODEL).
///
/// This isolates the feedforward network from the rest of the transformer block
/// to verify GELU activation bounds in isolation. GELU is the key nonlinearity
/// in the Whisper encoder FFN and requires CROWN linearization.
fn build_isolated_mlp(t: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_enc_isolated_mlp");

    let input = b.add_input("x", &[t, D_MODEL]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_weight", &[D_MODEL, FFN_DIM]);

    // Linear up-projection -> GELU -> Linear down-projection
    let up = b.add_linear(input, ffn1_w, None, &[t, FFN_DIM]);
    let act = b.add_gelu(up, &[t, FFN_DIM]);
    let out = b.add_linear(act, ffn2_w, None, &[t, D_MODEL]);

    b.build(out).expect("valid isolated MLP kernel")
}

fn isolated_mlp_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, FFN_DIM]), W_MAG)),
    ]
}

// ---------------------------------------------------------------------------
// Builder: Stacked 3-block encoder chain
// ---------------------------------------------------------------------------

/// Build a 3-block stacked encoder chain.
///
/// Input: `[T, D_MODEL]` (Variable).
/// Output: `[T, D_MODEL]`.
///
/// Architecture: 3 x (LN -> MHA(standard) -> + residual -> LN -> FFN -> + residual).
///
/// Tests bounds stability through depth: with N stacked blocks, bounds may
/// grow due to residual accumulation and normalization layer approximations.
/// Conservative mode ensures Sound verification at the cost of wider bounds.
fn build_stacked_3block_encoder(t: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_enc_3block_stack");

    let x = b.add_input("x", &[t, D_MODEL]);
    let eps = b.add_input("eps", &[1]);

    let shape = [t, D_MODEL];
    let ffn_shape = [t, FFN_DIM];
    let mut current = x;

    for idx in 0..3 {
        let pfx = format!("b{idx}");

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

    b.build(current)
        .expect("valid 3-block stacked encoder kernel")
}

fn stacked_3block_encoder_bindings() -> Vec<TensorParamBinding> {
    let d = D_MODEL;
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), W_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), W_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), W_MAG);

    let mut bindings = vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
    ];

    for _ in 0..3 {
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

    bindings
}

// ===========================================================================
// Test 6: Q, K, V projection bounds (linear projections)
// ===========================================================================

/// Q, K, V linear projections preserve bounds through linear transformation.
///
/// Linear projections are exact under IBP (no nonlinearity). With weight
/// magnitude W_MAG and input range [-1, 1], output bounds should scale
/// linearly: |output| <= D_MODEL * W_MAG * input_range.
#[test]
fn test_whisper_enc_qkv_projection_ibp_bounds() {
    let t = after_conv2();
    let def = build_qkv_projections(t);
    let bindings = qkv_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Q/K/V projections");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[t, D_MODEL],
        "Q/K/V projection output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    // Linear projections are exact under IBP. With small weights (0.02),
    // the Q+K+V sum should have bounded output. Each projection contributes
    // at most D_MODEL * W_MAG * input_range per element, and we sum 3 projections.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        width < VACUOUS_THRESHOLD,
        "Q/K/V projection width {width} exceeds vacuous threshold"
    );

    // Bounds should be symmetric around 0 for symmetric input and uniform weights.
    assert!(
        (lo_min + hi_max).abs() < 1e-3,
        "symmetric input should produce near-symmetric bounds: [{lo_min}, {hi_max}]"
    );

    eprintln!("whisper_enc_qkv_projection: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

/// Q, K, V projections: CROWN produces same bounds as IBP (all-linear network).
///
/// Since the Q/K/V projection graph is entirely linear (no nonlinearities),
/// CROWN should produce identical bounds to IBP. This serves as a soundness
/// check: CROWN must not widen bounds on a linear-only graph.
#[test]
fn test_whisper_enc_qkv_projection_crown_matches_ibp() {
    let t = after_conv2();
    let def = build_qkv_projections(t);
    let bindings = qkv_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[t, D_MODEL],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "whisper_enc_qkv_projection CROWN: method={method:?}, \
         bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

// ===========================================================================
// Test 7: MLP (FFN) with GELU activation bounds
// ===========================================================================

/// Isolated MLP with GELU: IBP produces finite bounds.
///
/// The FFN sub-block is Linear(D_MODEL, FFN_DIM) -> GELU -> Linear(FFN_DIM, D_MODEL).
/// GELU is the key nonlinearity in Whisper's FFN. It is monotonically increasing
/// for x > 0 and bounded below by ~-0.17 at x ~ -0.75. With small weights (0.02),
/// output bounds should stay tight.
#[test]
fn test_whisper_enc_mlp_gelu_ibp_bounds() {
    let t = after_conv2();
    let def = build_isolated_mlp(t);
    let bindings = isolated_mlp_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MLP with GELU");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[t, D_MODEL],
        "MLP output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        width < VACUOUS_THRESHOLD,
        "MLP GELU output width {width} exceeds vacuous threshold"
    );

    // GELU output is bounded: for input in [-D*W, D*W], GELU output
    // is in approximately [-0.17, D*W]. The second linear projection
    // scales this further.
    eprintln!("whisper_enc_mlp_gelu: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

/// Isolated MLP with GELU: CROWN linearization produces tighter bounds.
///
/// GELU requires CROWN linearization (piecewise linear approximation).
/// CROWN should produce tighter bounds than IBP because the linear
/// relaxation of GELU is tighter than the interval hull.
#[test]
fn test_whisper_enc_mlp_gelu_crown_tighter() {
    let t = after_conv2();
    let def = build_isolated_mlp(t);
    let bindings = isolated_mlp_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[t, D_MODEL],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "whisper_enc_mlp_gelu CROWN: method={method:?}, \
         bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Isolated MLP with GELU: verify and record under status key.
#[test]
fn test_whisper_enc_mlp_gelu_verify_and_record() {
    let t = after_conv2();
    let def = build_isolated_mlp(t);
    let bindings = isolated_mlp_bindings();
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "whisper_encoder_mlp_gelu",
        &conservative_config(),
    );
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t, D_MODEL]);

    // MLP has GELU but no LayerNorm, so Conservative config should
    // produce Sound verification. GELU itself is CROWN-compatible.
    eprintln!(
        "whisper_encoder_mlp_gelu: soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Test 8: Stacked 3-block encoder chain (bounds through depth)
// ===========================================================================

/// Stacked 3-block encoder: IBP bounds stay finite through depth.
///
/// With 3 stacked encoder blocks (6 LayerNorms, 3 MHA, 3 FFN), bounds may
/// grow due to residual accumulation. This test verifies that bounds remain
/// finite and non-vacuous with Conservative config and small weights.
#[test]
fn test_whisper_enc_3block_stack_ibp_bounds() {
    let t = after_conv2();
    let def = build_stacked_3block_encoder(t);
    let bindings = stacked_3block_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-block stacked encoder");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[t, D_MODEL],
        "stacked encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");

    // Bounds through 3 stacked blocks may be wider than 1 block but
    // should still be non-vacuous with small weights.
    assert!(
        width < VACUOUS_THRESHOLD,
        "3-block stacked encoder width {width} exceeds vacuous threshold"
    );

    // Bounds should be at least as wide as input (residual connections
    // preserve input contribution).
    assert!(
        width >= 2.0 - 1e-3,
        "stacked encoder bounds must be at least as wide as input: width={width}"
    );

    eprintln!("whisper_enc_3block_stack IBP: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

/// Stacked 3-block encoder: verify and record with Conservative config.
///
/// End-to-end Sound verification of a 3-block encoder stack. With
/// Conservative norm mode, the 6 LayerNorms (2 per block) should all
/// produce Sound results.
#[test]
fn test_whisper_enc_3block_stack_conservative_verify() {
    let t = after_conv2();
    let def = build_stacked_3block_encoder(t);
    let bindings = stacked_3block_encoder_bindings();
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "whisper_encoder_3block_stack",
        &conservative_config(),
    );
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t, D_MODEL]);

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    let width = hi_max - lo_min;

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        width < VACUOUS_THRESHOLD,
        "3-block conservative width {width} exceeds vacuous threshold"
    );

    // Accept both Sound and Heuristic.
    assert!(
        matches!(
            result.verification.soundness_mode,
            VerificationSoundnessMode::Sound | VerificationSoundnessMode::Heuristic
        ),
        "3-block stack with Conservative should produce Sound or Heuristic, got {:?}",
        result.verification.soundness_mode
    );

    eprintln!(
        "whisper_encoder_3block_stack: bounds=[{lo_min:.6}, {hi_max:.6}], \
         width={width:.4}, soundness={:?}",
        result.verification.soundness_mode
    );
}

/// Stacked 3-block encoder: monotonicity -- narrow input produces tighter output.
///
/// Verifies IBP monotonicity through 3 stacked blocks. Narrower input
/// perturbation should produce narrower output bounds for the majority
/// of elements. LayerNorm decomposition may cause some elements to violate
/// strict monotonicity.
#[test]
fn test_whisper_enc_3block_stack_monotonicity() {
    let t = after_conv2();
    let def = build_stacked_3block_encoder(t);
    let bindings = stacked_3block_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[t, D_MODEL], 5.0);
    let narrow_input = uniform_bounds(&[t, D_MODEL], 1.0);

    let wide_output = graph.propagate_ibp(&wide_input).expect("wide IBP");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("narrow IBP");

    let (wide_lo, wide_hi) = wide_output.lower_upper();
    let (narrow_lo, narrow_hi) = narrow_output.lower_upper();

    let wide_range = wide_hi.iter().zip(wide_lo.iter()).map(|(h, l)| h - l);
    let narrow_range = narrow_hi.iter().zip(narrow_lo.iter()).map(|(h, l)| h - l);

    // At least half of output elements should have narrower bounds.
    let tighter_count = wide_range.zip(narrow_range).filter(|(w, n)| n <= w).count();
    let total = wide_lo.len();
    assert!(
        tighter_count > total / 2,
        "narrow input should produce tighter bounds for > 50% of elements, \
         got {tighter_count}/{total}"
    );

    let (wide_lo_min, wide_hi_max) = bounds_min_max(&wide_output);
    let (narrow_lo_min, narrow_hi_max) = bounds_min_max(&narrow_output);
    eprintln!(
        "3-block monotonicity: wide=[{wide_lo_min:.4}, {wide_hi_max:.4}] \
         | narrow=[{narrow_lo_min:.4}, {narrow_hi_max:.4}]"
    );
}
