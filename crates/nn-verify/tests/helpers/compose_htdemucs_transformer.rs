// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HTDemucs transformer block compose verification tests (IBP).
//!
//! Tests NY IBP bounds propagation through the cross-attention
//! transformer bottleneck of the HTDemucs music source separation model.
//! HTDemucs uses a cross-domain transformer where temporal encoder output
//! attends to spectral encoder output (and vice versa).
//!
//! 1. **Self-attention bounds** — multi-head self-attention preserves IBP bounds.
//!
//! 2. **Cross-attention bounds** — cross-attention with different-modality KV
//!    preserves bounds (temporal queries, spectral KV).
//!
//! 3. **LayerNorm before attention** — pre-norm pattern maintains bounded output.
//!
//! 4. **FFN with GELU bounds** — 2-layer FFN with GELU activation preserves bounds.
//!
//! 5. **Residual connection bounds** — skip + branch bounded by sum of
//!    component bounds.
//!
//! 6. **Single transformer layer** — full layer (norm, self-attn, norm,
//!    cross-attn, norm, FFN) maintains bounds.
//!
//! 7. **Two-layer stack** — 2 sequential transformer layers maintain bounded
//!    outputs.
//!
//! 8. **Temporal-spectral cross-attention symmetry** — both cross-attention
//!    directions preserve comparable bounds.
//!
//! All tests use Conservative NormBoundsMode to target Sound classification.
//! Part of #4186.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::{
    AttentionMask, CrossAttentionBlockConfig, CrossAttentionBlockWeights, TransformerBlockConfig,
    TransformerBlockWeights,
};
use nn_verify::{
    tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerificationSoundnessMode,
    VerifyConfig,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Model dimension (matches bottleneck encoder output channels).
const D: usize = 8;

/// Number of attention heads. D must be divisible by NUM_HEADS.
const NUM_HEADS: usize = 2;

/// FFN intermediate dimension (2x model dim, standard ratio).
const FFN_DIM: usize = D * 2;

/// Temporal sequence length (encoder output time steps).
const T_SEQ: usize = 4;

/// Spectral sequence length (frequency bins from spectral encoder).
const F_SEQ: usize = 6;

/// Small weight magnitude for stable IBP propagation.
const WEIGHT_MAG: f32 = 0.01;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ---------------------------------------------------------------------------
// Binding helpers
// ---------------------------------------------------------------------------

fn push_weight(bindings: &mut Vec<TensorParamBinding>, shape: &[usize], val: f32) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(shape),
        val,
    )));
}

/// Add bindings for a self-attention TransformerBlockWeights (11 params).
fn add_self_attn_bindings(bindings: &mut Vec<TensorParamBinding>) {
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
    push_weight(bindings, &[D], 1.0); // ln1 gamma
    push_weight(bindings, &[D], 0.0); // ln1 beta
    push_weight(bindings, &[D], 1.0); // ln2 gamma
    push_weight(bindings, &[D], 0.0); // ln2 beta
    push_weight(bindings, &[D, D], WEIGHT_MAG); // Q
    push_weight(bindings, &[D, D], WEIGHT_MAG); // K
    push_weight(bindings, &[D, D], WEIGHT_MAG); // V
    push_weight(bindings, &[D, D], WEIGHT_MAG); // out
    push_weight(bindings, &[FFN_DIM, D], WEIGHT_MAG); // ffn1
    push_weight(bindings, &[D, FFN_DIM], WEIGHT_MAG); // ffn2
}

/// Add bindings for a CrossAttentionBlockWeights (15 params).
fn add_cross_attn_bindings(bindings: &mut Vec<TensorParamBinding>) {
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
    push_weight(bindings, &[D], 1.0); // ln1 gamma (Q branch)
    push_weight(bindings, &[D], 0.0); // ln1 beta
    push_weight(bindings, &[D], 1.0); // ln2 gamma (KV branch)
    push_weight(bindings, &[D], 0.0); // ln2 beta
    push_weight(bindings, &[D], 1.0); // ln3 gamma (pre-FFN)
    push_weight(bindings, &[D], 0.0); // ln3 beta
    push_weight(bindings, &[D], 1.0); // ln_out gamma
    push_weight(bindings, &[D], 0.0); // ln_out beta
    push_weight(bindings, &[D, D], WEIGHT_MAG); // Q
    push_weight(bindings, &[D, D], WEIGHT_MAG); // K
    push_weight(bindings, &[D, D], WEIGHT_MAG); // V
    push_weight(bindings, &[D, D], WEIGHT_MAG); // out
    push_weight(bindings, &[FFN_DIM, D], WEIGHT_MAG); // ffn1
    push_weight(bindings, &[D, FFN_DIM], WEIGHT_MAG); // ffn2
}

// ---------------------------------------------------------------------------
// Input node builders
// ---------------------------------------------------------------------------

/// Add TransformerBlockWeights input nodes to the builder.
fn add_self_attn_weights(b: &mut TensorBlockBuilder, prefix: &str) -> TransformerBlockWeights {
    let eps = b.add_input(&format!("{prefix}_eps"), &[1]);
    TransformerBlockWeights {
        ln1_weight: b.add_input(&format!("{prefix}_ln1_w"), &[D]),
        ln1_bias: b.add_input(&format!("{prefix}_ln1_b"), &[D]),
        ln2_weight: b.add_input(&format!("{prefix}_ln2_w"), &[D]),
        ln2_bias: b.add_input(&format!("{prefix}_ln2_b"), &[D]),
        q_weight: b.add_input(&format!("{prefix}_q_w"), &[D, D]),
        k_weight: b.add_input(&format!("{prefix}_k_w"), &[D, D]),
        v_weight: b.add_input(&format!("{prefix}_v_w"), &[D, D]),
        out_weight: b.add_input(&format!("{prefix}_out_w"), &[D, D]),
        ffn1_weight: b.add_input(&format!("{prefix}_ffn1_w"), &[FFN_DIM, D]),
        ffn2_weight: b.add_input(&format!("{prefix}_ffn2_w"), &[D, FFN_DIM]),
        eps,
    }
}

/// Add CrossAttentionBlockWeights input nodes to the builder.
fn add_cross_attn_weights(b: &mut TensorBlockBuilder, prefix: &str) -> CrossAttentionBlockWeights {
    let eps = b.add_input(&format!("{prefix}_eps"), &[1]);
    CrossAttentionBlockWeights {
        ln1_weight: b.add_input(&format!("{prefix}_ln1_w"), &[D]),
        ln1_bias: b.add_input(&format!("{prefix}_ln1_b"), &[D]),
        ln2_weight: b.add_input(&format!("{prefix}_ln2_w"), &[D]),
        ln2_bias: b.add_input(&format!("{prefix}_ln2_b"), &[D]),
        ln3_weight: b.add_input(&format!("{prefix}_ln3_w"), &[D]),
        ln3_bias: b.add_input(&format!("{prefix}_ln3_b"), &[D]),
        ln_out_weight: b.add_input(&format!("{prefix}_lnout_w"), &[D]),
        ln_out_bias: b.add_input(&format!("{prefix}_lnout_b"), &[D]),
        q_weight: b.add_input(&format!("{prefix}_q_w"), &[D, D]),
        k_weight: b.add_input(&format!("{prefix}_k_w"), &[D, D]),
        v_weight: b.add_input(&format!("{prefix}_v_w"), &[D, D]),
        out_weight: b.add_input(&format!("{prefix}_out_w"), &[D, D]),
        ffn1_weight: b.add_input(&format!("{prefix}_ffn1_w"), &[FFN_DIM, D]),
        ffn2_weight: b.add_input(&format!("{prefix}_ffn2_w"), &[D, FFN_DIM]),
        eps,
    }
}

// ===========================================================================
// 1. Self-attention bounds
// ===========================================================================

/// Build multi-head self-attention on temporal input [T, D].
fn build_self_attention() -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("htdemucs_tf_self_attn");
    let data = b.add_input("data", &[T_SEQ, D]);
    let weights = add_self_attn_weights(&mut b, "sa");

    let tc = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };
    let output = b
        .add_transformer_block(data, &weights, &tc)
        .expect("self-attention block");

    let def = b.build(output).expect("valid self-attention");
    let mut bindings = vec![TensorParamBinding::Variable];
    add_self_attn_bindings(&mut bindings);
    (def, bindings)
}

#[test]
fn test_htdemucs_tf_self_attn_def_validates() {
    let (def, _) = build_self_attention();
    def.validate().expect("self-attention should validate");
}

#[test]
fn test_htdemucs_tf_self_attn_ibp() {
    let (def, bindings) = build_self_attention();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through self-attention");
    assert_eq!(output.lower_upper().0.shape(), &[T_SEQ, D]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs transformer self-attention IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower bound finite, got {lo}");
    assert!(hi.is_finite(), "upper bound finite, got {hi}");
    assert!(lo.abs() < 1e6, "lower bound < 1e6, got {lo}");
    assert!(hi.abs() < 1e6, "upper bound < 1e6, got {hi}");
}

#[test]
fn test_htdemucs_tf_self_attn_conservative_sound() {
    let (def, bindings) = build_self_attention();
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_tf_self_attn",
        &conservative_config(),
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs tf self-attention (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 2. Cross-attention bounds
// ===========================================================================

/// Build cross-attention: temporal queries [T, D], spectral KV [F, D].
fn build_cross_attention() -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("htdemucs_tf_cross_attn");
    let q_input = b.add_input("temporal", &[T_SEQ, D]);
    let kv_input = b.add_input("spectral", &[F_SEQ, D]);
    let weights = add_cross_attn_weights(&mut b, "ca");

    let cac = CrossAttentionBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };
    let output = b
        .add_cross_attention_transformer_block(q_input, kv_input, &weights, &cac)
        .expect("cross-attention block");

    let def = b.build(output).expect("valid cross-attention");
    // Variable for temporal Q input; constant for spectral KV.
    let mut bindings = vec![TensorParamBinding::Variable];
    push_weight(&mut bindings, &[F_SEQ, D], 0.5);
    add_cross_attn_bindings(&mut bindings);
    (def, bindings)
}

#[test]
fn test_htdemucs_tf_cross_attn_def_validates() {
    let (def, _) = build_cross_attention();
    def.validate().expect("cross-attention should validate");
}

#[test]
fn test_htdemucs_tf_cross_attn_ibp() {
    let (def, bindings) = build_cross_attention();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attention");
    assert_eq!(output.lower_upper().0.shape(), &[T_SEQ, D]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs transformer cross-attention IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower bound finite, got {lo}");
    assert!(hi.is_finite(), "upper bound finite, got {hi}");
    assert!(lo.abs() < 1e6, "lower bound < 1e6, got {lo}");
    assert!(hi.abs() < 1e6, "upper bound < 1e6, got {hi}");
}

#[test]
fn test_htdemucs_tf_cross_attn_conservative_sound() {
    let (def, bindings) = build_cross_attention();
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_tf_cross_attn",
        &conservative_config(),
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs tf cross-attention (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 3. LayerNorm before attention
// ===========================================================================

/// Build isolated pre-norm pattern: LayerNorm → Linear → GELU.
/// Tests that LayerNorm normalization produces bounded output suitable
/// for downstream attention.
fn build_layernorm_pre_attn() -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("htdemucs_tf_layernorm_pre_attn");
    let data = b.add_input("data", &[T_SEQ, D]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D]);
    let ln_b = b.add_input("ln_b", &[D]);
    let proj_w = b.add_input("proj_w", &[D, D]);

    let normed = b.add_layer_norm(data, eps, 1, ln_w, ln_b, &[T_SEQ, D]);
    let output = b.add_linear(normed, proj_w, None, &[T_SEQ, D]);

    let def = b.build(output).expect("valid layernorm pre-attn");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D, D]), WEIGHT_MAG)),
    ];
    (def, bindings)
}

#[test]
fn test_htdemucs_tf_layernorm_pre_attn_def_validates() {
    let (def, _) = build_layernorm_pre_attn();
    def.validate().expect("layernorm pre-attn should validate");
}

#[test]
fn test_htdemucs_tf_layernorm_pre_attn_ibp() {
    let (def, bindings) = build_layernorm_pre_attn();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through layernorm pre-attn");
    assert_eq!(output.lower_upper().0.shape(), &[T_SEQ, D]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs tf LayerNorm pre-attn IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower bound finite, got {lo}");
    assert!(hi.is_finite(), "upper bound finite, got {hi}");
}

#[test]
fn test_htdemucs_tf_layernorm_pre_attn_conservative_sound() {
    let (def, bindings) = build_layernorm_pre_attn();
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_tf_layernorm_pre_attn",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs tf LayerNorm pre-attn (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 4. FFN with GELU bounds
// ===========================================================================

/// Build isolated FFN: Linear(D -> FFN_DIM) → GELU → Linear(FFN_DIM -> D).
/// This is the feed-forward sub-block within each transformer layer.
fn build_ffn_gelu() -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("htdemucs_tf_ffn_gelu");
    let data = b.add_input("data", &[T_SEQ, D]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D]);
    let ffn2_w = b.add_input("ffn2_w", &[D, FFN_DIM]);

    let ffn1 = b.add_linear(data, ffn1_w, None, &[T_SEQ, FFN_DIM]);
    let act = b.add_gelu(ffn1, &[T_SEQ, FFN_DIM]);
    let output = b.add_linear(act, ffn2_w, None, &[T_SEQ, D]);

    let def = b.build(output).expect("valid FFN GELU");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D, FFN_DIM]), WEIGHT_MAG)),
    ];
    (def, bindings)
}

#[test]
fn test_htdemucs_tf_ffn_gelu_def_validates() {
    let (def, _) = build_ffn_gelu();
    def.validate().expect("FFN GELU should validate");
}

#[test]
fn test_htdemucs_tf_ffn_gelu_ibp() {
    let (def, bindings) = build_ffn_gelu();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through FFN GELU");
    assert_eq!(output.lower_upper().0.shape(), &[T_SEQ, D]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs tf FFN GELU IBP: [{lo}, {hi}]");
    // Small weights with GELU activation should produce moderate bounds
    assert!(lo.abs() < 100.0, "lower bound < 100, got {lo}");
    assert!(hi.abs() < 100.0, "upper bound < 100, got {hi}");
}

#[test]
fn test_htdemucs_tf_ffn_gelu_crown() {
    let (def, bindings) = build_ffn_gelu();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[T_SEQ, D]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs tf FFN GELU: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 5. Residual connection bounds
// ===========================================================================

/// Build residual connection: input + Linear(GELU(Linear(input))).
/// Verifies that the skip connection bounds the output by the sum of
/// identity range and branch range.
fn build_residual_connection() -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("htdemucs_tf_residual");
    let data = b.add_input("data", &[T_SEQ, D]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D]);
    let ffn2_w = b.add_input("ffn2_w", &[D, FFN_DIM]);

    // Branch: Linear → GELU → Linear
    let ffn1 = b.add_linear(data, ffn1_w, None, &[T_SEQ, FFN_DIM]);
    let act = b.add_gelu(ffn1, &[T_SEQ, FFN_DIM]);
    let branch = b.add_linear(act, ffn2_w, None, &[T_SEQ, D]);

    // Residual: input + branch
    let output = b.add_binary_add(data, branch, &[T_SEQ, D]);

    let def = b.build(output).expect("valid residual connection");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D, FFN_DIM]), WEIGHT_MAG)),
    ];
    (def, bindings)
}

#[test]
fn test_htdemucs_tf_residual_def_validates() {
    let (def, _) = build_residual_connection();
    def.validate().expect("residual connection should validate");
}

#[test]
fn test_htdemucs_tf_residual_ibp() {
    let (def, bindings) = build_residual_connection();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through residual connection");
    assert_eq!(output.lower_upper().0.shape(), &[T_SEQ, D]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs tf residual IBP: [{lo}, {hi}]");
    // Residual = input + branch. With small weights the branch contribution
    // is small, so output bounds should be close to input bounds.
    assert!(lo >= -10.0, "residual lower should be >= -10, got {lo}");
    assert!(hi <= 10.0, "residual upper should be <= 10, got {hi}");
}

#[test]
fn test_htdemucs_tf_residual_conservative_sound() {
    let (def, bindings) = build_residual_connection();
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_tf_residual",
        &conservative_config(),
    );

    // No normalization layers in the residual block alone, so Sound expected.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Residual (no norms) should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs tf residual (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 6. Single transformer layer
// ===========================================================================

/// Build a full HTDemucs transformer layer:
/// self-attention → cross-attention (temporal queries spectral KV).
/// This is the core bottleneck block between encoder and decoder.
fn build_single_transformer_layer() -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>)
{
    let mut b = TensorBlockBuilder::new("htdemucs_tf_single_layer");
    let temporal = b.add_input("temporal", &[T_SEQ, D]);
    let spectral = b.add_input("spectral", &[F_SEQ, D]);

    // Self-attention on temporal sequence
    let sa_weights = add_self_attn_weights(&mut b, "sa");
    let tc = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };
    let after_sa = b
        .add_transformer_block(temporal, &sa_weights, &tc)
        .expect("self-attention");

    // Cross-attention: temporal queries spectral KV
    let ca_weights = add_cross_attn_weights(&mut b, "ca");
    let cac = CrossAttentionBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };
    let output = b
        .add_cross_attention_transformer_block(after_sa, spectral, &ca_weights, &cac)
        .expect("cross-attention");

    let def = b.build(output).expect("valid single transformer layer");

    let mut bindings = vec![TensorParamBinding::Variable]; // temporal
    push_weight(&mut bindings, &[F_SEQ, D], 0.5); // spectral (constant)
    add_self_attn_bindings(&mut bindings);
    add_cross_attn_bindings(&mut bindings);
    (def, bindings)
}

#[test]
fn test_htdemucs_tf_single_layer_def_validates() {
    let (def, _) = build_single_transformer_layer();
    def.validate()
        .expect("single transformer layer should validate");
}

#[test]
fn test_htdemucs_tf_single_layer_ibp() {
    let (def, bindings) = build_single_transformer_layer();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through single transformer layer");
    assert_eq!(output.lower_upper().0.shape(), &[T_SEQ, D]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs tf single layer IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower bound finite, got {lo}");
    assert!(hi.is_finite(), "upper bound finite, got {hi}");
    assert!(lo.abs() < 1e6, "single layer lower bound < 1e6, got {lo}");
    assert!(hi.abs() < 1e6, "single layer upper bound < 1e6, got {hi}");
}

#[test]
fn test_htdemucs_tf_single_layer_conservative_sound() {
    let (def, bindings) = build_single_transformer_layer();
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_tf_single_layer",
        &conservative_config(),
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs tf single layer (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

#[test]
fn test_htdemucs_tf_single_layer_crown() {
    let (def, bindings) = build_single_transformer_layer();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[T_SEQ, D]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs tf single layer: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 7. Two-layer stack
// ===========================================================================

/// Build 2 sequential transformer layers (self-attn + cross-attn each).
/// Tests bounds stability through repeated transformer processing.
fn build_two_layer_stack() -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("htdemucs_tf_two_layer_stack");
    let temporal = b.add_input("temporal", &[T_SEQ, D]);
    let spectral = b.add_input("spectral", &[F_SEQ, D]);

    let tc = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };
    let cac = CrossAttentionBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };

    // Layer 1
    let sa1_w = add_self_attn_weights(&mut b, "l1_sa");
    let ca1_w = add_cross_attn_weights(&mut b, "l1_ca");
    let after_sa1 = b
        .add_transformer_block(temporal, &sa1_w, &tc)
        .expect("layer 1 self-attn");
    let after_ca1 = b
        .add_cross_attention_transformer_block(after_sa1, spectral, &ca1_w, &cac)
        .expect("layer 1 cross-attn");

    // Layer 2
    let sa2_w = add_self_attn_weights(&mut b, "l2_sa");
    let ca2_w = add_cross_attn_weights(&mut b, "l2_ca");
    let after_sa2 = b
        .add_transformer_block(after_ca1, &sa2_w, &tc)
        .expect("layer 2 self-attn");
    let output = b
        .add_cross_attention_transformer_block(after_sa2, spectral, &ca2_w, &cac)
        .expect("layer 2 cross-attn");

    let def = b.build(output).expect("valid two-layer stack");

    let mut bindings = vec![TensorParamBinding::Variable]; // temporal
    push_weight(&mut bindings, &[F_SEQ, D], 0.5); // spectral (constant)
                                                  // Layer 1 bindings
    add_self_attn_bindings(&mut bindings);
    add_cross_attn_bindings(&mut bindings);
    // Layer 2 bindings
    add_self_attn_bindings(&mut bindings);
    add_cross_attn_bindings(&mut bindings);
    (def, bindings)
}

#[test]
fn test_htdemucs_tf_two_layer_stack_def_validates() {
    let (def, _) = build_two_layer_stack();
    def.validate().expect("two-layer stack should validate");
}

#[test]
fn test_htdemucs_tf_two_layer_stack_ibp() {
    let (def, bindings) = build_two_layer_stack();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through two-layer stack");
    assert_eq!(output.lower_upper().0.shape(), &[T_SEQ, D]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs tf two-layer stack IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower bound finite, got {lo}");
    assert!(hi.is_finite(), "upper bound finite, got {hi}");
    // Two layers may widen bounds but should remain tractable
    assert!(
        lo.abs() < 1e8,
        "two-layer stack lower bound < 1e8, got {lo}"
    );
    assert!(
        hi.abs() < 1e8,
        "two-layer stack upper bound < 1e8, got {hi}"
    );
}

#[test]
fn test_htdemucs_tf_two_layer_stack_conservative_sound() {
    let (def, bindings) = build_two_layer_stack();
    let input = uniform_bounds(&[T_SEQ, D], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_tf_two_layer_stack",
        &conservative_config(),
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs tf two-layer stack (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 8. Temporal-spectral cross-attention symmetry
// ===========================================================================

/// Build both cross-attention directions:
/// - temporal→spectral: temporal queries, spectral KV
/// - spectral→temporal: spectral queries, temporal KV
///
/// Verifies that both directions produce finite, comparable-magnitude bounds
/// (the architecture should be symmetric modulo sequence length differences).
fn build_bidirectional_cross_attn() -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    nn_dsl::tensor_ir::TensorKernelDef,
    Vec<TensorParamBinding>,
    Vec<TensorParamBinding>,
) {
    // Direction 1: temporal queries spectral KV
    let mut b1 = TensorBlockBuilder::new("htdemucs_tf_t2s_cross_attn");
    let t_input = b1.add_input("temporal", &[T_SEQ, D]);
    let s_kv = b1.add_input("spectral_kv", &[F_SEQ, D]);
    let ca1_w = add_cross_attn_weights(&mut b1, "t2s");
    let cac = CrossAttentionBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };
    let t2s_out = b1
        .add_cross_attention_transformer_block(t_input, s_kv, &ca1_w, &cac)
        .expect("t2s cross-attn");
    let def1 = b1.build(t2s_out).expect("valid t2s");

    let mut bindings1 = vec![TensorParamBinding::Variable]; // temporal
    push_weight(&mut bindings1, &[F_SEQ, D], 0.5); // spectral (constant)
    add_cross_attn_bindings(&mut bindings1);

    // Direction 2: spectral queries temporal KV
    let mut b2 = TensorBlockBuilder::new("htdemucs_tf_s2t_cross_attn");
    let s_input = b2.add_input("spectral", &[F_SEQ, D]);
    let t_kv = b2.add_input("temporal_kv", &[T_SEQ, D]);
    let ca2_w = add_cross_attn_weights(&mut b2, "s2t");
    let s2t_out = b2
        .add_cross_attention_transformer_block(s_input, t_kv, &ca2_w, &cac)
        .expect("s2t cross-attn");
    let def2 = b2.build(s2t_out).expect("valid s2t");

    let mut bindings2 = vec![TensorParamBinding::Variable]; // spectral
    push_weight(&mut bindings2, &[T_SEQ, D], 0.5); // temporal (constant)
    add_cross_attn_bindings(&mut bindings2);

    (def1, def2, bindings1, bindings2)
}

#[test]
fn test_htdemucs_tf_bidirectional_cross_attn_def_validates() {
    let (def1, def2, _, _) = build_bidirectional_cross_attn();
    def1.validate().expect("t2s should validate");
    def2.validate().expect("s2t should validate");
}

#[test]
fn test_htdemucs_tf_bidirectional_cross_attn_ibp() {
    let (def1, def2, bindings1, bindings2) = build_bidirectional_cross_attn();

    // Direction 1: temporal → spectral
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("t2s graph");
    let t_input = uniform_bounds(&[T_SEQ, D], 1.0);
    let t2s_output = graph1.propagate_ibp(&t_input).expect("t2s IBP");
    assert_eq!(t2s_output.lower_upper().0.shape(), &[T_SEQ, D]);
    assert_bounds_valid(&t2s_output);

    let (t2s_lo, t2s_hi) = bounds_min_max(&t2s_output);
    eprintln!("HTDemucs temporal→spectral cross-attn IBP: [{t2s_lo}, {t2s_hi}]");

    // Direction 2: spectral → temporal
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("s2t graph");
    let s_input = uniform_bounds(&[F_SEQ, D], 1.0);
    let s2t_output = graph2.propagate_ibp(&s_input).expect("s2t IBP");
    assert_eq!(s2t_output.lower_upper().0.shape(), &[F_SEQ, D]);
    assert_bounds_valid(&s2t_output);

    let (s2t_lo, s2t_hi) = bounds_min_max(&s2t_output);
    eprintln!("HTDemucs spectral→temporal cross-attn IBP: [{s2t_lo}, {s2t_hi}]");

    // Both directions should have comparable magnitude bounds
    // (same weights, same architecture, just different sequence lengths)
    let t2s_range = t2s_hi - t2s_lo;
    let s2t_range = s2t_hi - s2t_lo;
    eprintln!(
        "Range comparison: t2s={t2s_range:.4}, s2t={s2t_range:.4}, ratio={:.4}",
        if s2t_range > 0.0 {
            t2s_range / s2t_range
        } else {
            f32::INFINITY
        }
    );

    // Both directions must be finite with moderate magnitude
    assert!(
        t2s_lo.abs() < 1e6 && t2s_hi.abs() < 1e6,
        "t2s bounds should be < 1e6, got [{t2s_lo}, {t2s_hi}]"
    );
    assert!(
        s2t_lo.abs() < 1e6 && s2t_hi.abs() < 1e6,
        "s2t bounds should be < 1e6, got [{s2t_lo}, {s2t_hi}]"
    );

    // Symmetry: ranges should be within 100x of each other
    // (IBP widening may differ with sequence length, so generous tolerance)
    if t2s_range > 1e-10 && s2t_range > 1e-10 {
        let ratio = (t2s_range / s2t_range).max(s2t_range / t2s_range);
        assert!(
            ratio < 100.0,
            "cross-attention directions should have comparable range, got ratio {ratio:.2}"
        );
    }
}

#[test]
fn test_htdemucs_tf_bidirectional_cross_attn_conservative_sound() {
    let (def1, def2, bindings1, bindings2) = build_bidirectional_cross_attn();

    // Direction 1
    let t_input = uniform_bounds(&[T_SEQ, D], 1.0);
    let result1 = verify_and_assert_with_config(
        &def1,
        &bindings1,
        &t_input,
        "htdemucs_tf_t2s_cross_attn",
        &conservative_config(),
    );
    let (lo1, hi1) = bounds_min_max(&result1.output_bounds);
    eprintln!(
        "HTDemucs t2s (Conservative): bounds=[{lo1}, {hi1}], soundness={:?}",
        result1.verification.soundness_mode
    );

    // Direction 2
    let s_input = uniform_bounds(&[F_SEQ, D], 1.0);
    let result2 = verify_and_assert_with_config(
        &def2,
        &bindings2,
        &s_input,
        "htdemucs_tf_s2t_cross_attn",
        &conservative_config(),
    );
    let (lo2, hi2) = bounds_min_max(&result2.output_bounds);
    eprintln!(
        "HTDemucs s2t (Conservative): bounds=[{lo2}, {hi2}], soundness={:?}",
        result2.verification.soundness_mode
    );
}
