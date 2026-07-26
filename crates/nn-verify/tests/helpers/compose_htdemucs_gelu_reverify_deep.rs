// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep HTDemucs GeLU re-verification after NY CROWN relaxation fix.
//!
//! Extends `compose_htdemucs_gelu_reverify.rs` with deeper composition patterns:
//! two-stage encoder cascade, multi-DConv stack, GeLU-erf transformer FFN,
//! and encoder+DConv composite stage.
//!
//! Part of #4314: Re-verify GeLU models after NY CROWN relaxation fix.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    uniform_bounds, verify_and_assert, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerifyConfig};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

/// Input audio channels.
const IN_CH: usize = 4;
/// First encoder stage output channels.
const ENC1_CH: usize = 8;
/// Second encoder stage output channels (doubles per stage in HTDemucs).
const ENC2_CH: usize = 16;
/// Temporal input length.
const T_IN: usize = 32;
/// Encoder Conv1d kernel size (production HTDemucs = 8).
const ENC_KERNEL: usize = 8;
/// Encoder Conv1d stride (production HTDemucs = 4).
const ENC_STRIDE: usize = 4;
/// Encoder Conv1d padding (production HTDemucs = kernel/4 = 2).
const ENC_PADDING: usize = 2;
/// Transformer model dimension.
const D_MODEL: usize = 8;
/// Transformer sequence length.
const T_SEQ: usize = 4;
/// FFN intermediate dimension (2x model dim).
const FFN_DIM: usize = D_MODEL * 2;
/// Small weight magnitude.
const W_MAG: f32 = 0.01;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

fn enc1_t_out() -> usize {
    conv1d_out_len(T_IN, ENC_KERNEL, ENC_STRIDE, ENC_PADDING)
}

fn enc2_t_out() -> usize {
    conv1d_out_len(enc1_t_out(), ENC_KERNEL, ENC_STRIDE, ENC_PADDING)
}

// ===========================================================================
// 1. Two-stage encoder cascade: Conv+GeLU -> Conv+GeLU
// ===========================================================================

/// Build two cascaded encoder stages:
///   Stage 1: Conv1d(IN_CH -> ENC1_CH, k=8, s=4) -> GeLU
///   Stage 2: Conv1d(ENC1_CH -> ENC2_CH, k=8, s=4) -> GeLU
fn build_two_stage_encoder() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let t1 = enc1_t_out();
    let t2 = enc2_t_out();
    let mut b = TensorBlockBuilder::new("htdemucs_two_stage_enc_gelu_reverify");

    let data = b.add_input("data", &[IN_CH, T_IN]);
    let c1_w = b.add_input("c1_w", &[ENC1_CH, IN_CH, ENC_KERNEL]);
    let c1_b = b.add_input("c1_b", &[ENC1_CH]);
    let c1 = b.add_conv1d(
        data,
        c1_w,
        Some(c1_b),
        ENC_STRIDE,
        ENC_PADDING,
        &[ENC1_CH, t1],
    );
    let g1 = b.add_gelu(c1, &[ENC1_CH, t1]);

    let c2_w = b.add_input("c2_w", &[ENC2_CH, ENC1_CH, ENC_KERNEL]);
    let c2_b = b.add_input("c2_b", &[ENC2_CH]);
    let c2 = b.add_conv1d(
        g1,
        c2_w,
        Some(c2_b),
        ENC_STRIDE,
        ENC_PADDING,
        &[ENC2_CH, t2],
    );
    let out = b.add_gelu(c2, &[ENC2_CH, t2]);

    let def = b.build(out).expect("valid two-stage encoder");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ENC1_CH, IN_CH, ENC_KERNEL]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC1_CH]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ENC2_CH, ENC1_CH, ENC_KERNEL]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC2_CH]), 0.0f32)),
    ];
    (def, bindings)
}

#[test]
fn test_htdemucs_two_stage_enc_gelu_crown_after_fix() {
    let (def, bindings) = build_two_stage_encoder();
    let t2 = enc2_t_out();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(ibp_output.lower_upper().0.shape(), &[ENC2_CH, t2]);
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    eprintln!(
        "HTDemucs two-stage encoder+GeLU IBP: [{ibp_lo}, {ibp_hi}], width={:.4}",
        ibp_hi - ibp_lo
    );

    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(crown_output.lower_upper().0.shape(), &[ENC2_CH, t2]);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!(
        "HTDemucs two-stage encoder+GeLU CROWN: method={method:?}, \
         [{crown_lo}, {crown_hi}], width={:.4}",
        crown_hi - crown_lo
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    assert!(ibp_lo.abs() < 100.0, "IBP lower < 100, got {ibp_lo}");
    assert!(ibp_hi.abs() < 100.0, "IBP upper < 100, got {ibp_hi}");
}

// ===========================================================================
// 2. Multi-DConv stack: 2 DConv blocks with residual
// ===========================================================================

/// Build two stacked DConv sub-layers with residual.
fn build_multi_dconv_gelu() -> (TensorKernelDef, usize, Vec<TensorParamBinding>) {
    let ch = ENC1_CH;
    let t = enc1_t_out();
    let mut b = TensorBlockBuilder::new("htdemucs_multi_dconv_gelu_reverify");

    let x = b.add_input("x", &[ch, t]);
    let shape = [ch, t];

    // DConv block 1
    let dc1_w = b.add_input("dc1_w", &[ch, ch, 3]);
    let dc1_b = b.add_input("dc1_b", &[ch]);
    let dc1 = b.add_conv1d(x, dc1_w, Some(dc1_b), 1, 1, &shape);
    let gn1_eps = b.add_input("gn1_eps", &[1]);
    let gn1_g = b.add_input("gn1_gamma", &[ch]);
    let gn1_b = b.add_input("gn1_beta", &[ch]);
    let n1 = b.add_group_norm_g1(dc1, gn1_eps, Some(gn1_g), Some(gn1_b), ch, t);
    let a1 = b.add_gelu(n1, &shape);
    let pw1_w = b.add_input("pw1_w", &[ch, ch, 1]);
    let pw1_b = b.add_input("pw1_b", &[ch]);
    let pw1 = b.add_conv1d(a1, pw1_w, Some(pw1_b), 1, 0, &shape);
    let r1 = b.add_binary_add(x, pw1, &shape);

    // DConv block 2
    let dc2_w = b.add_input("dc2_w", &[ch, ch, 3]);
    let dc2_b = b.add_input("dc2_b", &[ch]);
    let dc2 = b.add_conv1d(r1, dc2_w, Some(dc2_b), 1, 1, &shape);
    let gn2_eps = b.add_input("gn2_eps", &[1]);
    let gn2_g = b.add_input("gn2_gamma", &[ch]);
    let gn2_b = b.add_input("gn2_beta", &[ch]);
    let n2 = b.add_group_norm_g1(dc2, gn2_eps, Some(gn2_g), Some(gn2_b), ch, t);
    let a2 = b.add_gelu(n2, &shape);
    let pw2_w = b.add_input("pw2_w", &[ch, ch, 1]);
    let pw2_b = b.add_input("pw2_b", &[ch]);
    let pw2 = b.add_conv1d(a2, pw2_w, Some(pw2_b), 1, 0, &shape);
    let out = b.add_binary_add(r1, pw2, &shape);

    let def = b.build(out).expect("valid multi-DConv+GeLU kernel");

    let dconv_block = || -> Vec<TensorParamBinding> {
        vec![
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch, ch, 3]), W_MAG)),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 1.0f32)),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch, ch, 1]), W_MAG)),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
        ]
    };

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(dconv_block());
    bindings.extend(dconv_block());
    (def, t, bindings)
}

#[test]
fn test_htdemucs_multi_dconv_gelu_crown_after_fix() {
    let (def, t, bindings) = build_multi_dconv_gelu();
    let ch = ENC1_CH;
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[ch, t], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    eprintln!(
        "HTDemucs multi-DConv (2 blocks) GeLU IBP: [{ibp_lo}, {ibp_hi}], width={:.4}",
        ibp_hi - ibp_lo
    );

    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!(
        "HTDemucs multi-DConv (2 blocks) GeLU CROWN: method={method:?}, \
         [{crown_lo}, {crown_hi}], width={:.4}",
        crown_hi - crown_lo
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }
}

// ===========================================================================
// 3. GeLU-erf in transformer FFN
// ===========================================================================

/// Build transformer FFN with GeLU-erf: LayerNorm -> Linear -> GeLU-erf -> Linear -> residual.
fn build_transformer_ffn_gelu_erf() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("htdemucs_xformer_ffn_gelu_erf_reverify");

    let x = b.add_input("x", &[T_SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let shape = [T_SEQ, D_MODEL];
    let ffn_shape = [T_SEQ, FFN_DIM];
    let normed = b.add_layer_norm(x, eps, 1, ln_w, ln_b, &shape);
    let h = b.add_linear(normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu_erf(h, &ffn_shape);
    let proj = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(x, proj, &shape);

    let def = b.build(out).expect("valid FFN+GeLU-erf kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, FFN_DIM]), W_MAG)),
    ];
    (def, bindings)
}

#[test]
fn test_htdemucs_xformer_ffn_gelu_erf_crown_after_fix() {
    let (def, bindings) = build_transformer_ffn_gelu_erf();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[T_SEQ, D_MODEL], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    eprintln!(
        "HTDemucs transformer FFN+GeLU-erf IBP: [{ibp_lo}, {ibp_hi}], width={:.4}",
        ibp_hi - ibp_lo
    );

    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!(
        "HTDemucs transformer FFN+GeLU-erf CROWN: method={method:?}, \
         [{crown_lo}, {crown_hi}], width={:.4}",
        crown_hi - crown_lo
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }
    assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
    assert!(crown_lo.is_finite() && crown_hi.is_finite());
}

// ===========================================================================
// 4. Encoder+DConv composite: full single encoder stage
// ===========================================================================

/// Build a complete encoder stage: Conv1d+GeLU downsampling + DConv+GroupNorm+GeLU.
fn build_enc_stage_with_dconv() -> (TensorKernelDef, usize, Vec<TensorParamBinding>) {
    let t1 = enc1_t_out();
    let ch = ENC1_CH;
    let mut b = TensorBlockBuilder::new("htdemucs_enc_stage_dconv_gelu_reverify");

    let data = b.add_input("data", &[IN_CH, T_IN]);
    let c_w = b.add_input("c_w", &[ch, IN_CH, ENC_KERNEL]);
    let c_b = b.add_input("c_b", &[ch]);
    let conv = b.add_conv1d(data, c_w, Some(c_b), ENC_STRIDE, ENC_PADDING, &[ch, t1]);
    let enc_act = b.add_gelu(conv, &[ch, t1]);

    let dc_w = b.add_input("dc_w", &[ch, ch, 3]);
    let dc_b = b.add_input("dc_b", &[ch]);
    let dc = b.add_conv1d(enc_act, dc_w, Some(dc_b), 1, 1, &[ch, t1]);
    let gn_eps = b.add_input("gn_eps", &[1]);
    let gn_g = b.add_input("gn_g", &[ch]);
    let gn_b = b.add_input("gn_b", &[ch]);
    let normed = b.add_group_norm_g1(dc, gn_eps, Some(gn_g), Some(gn_b), ch, t1);
    let dconv_act = b.add_gelu(normed, &[ch, t1]);
    let pw_w = b.add_input("pw_w", &[ch, ch, 1]);
    let pw_b = b.add_input("pw_b", &[ch]);
    let pw = b.add_conv1d(dconv_act, pw_w, Some(pw_b), 1, 0, &[ch, t1]);
    let out = b.add_binary_add(enc_act, pw, &[ch, t1]);

    let def = b.build(out).expect("valid encoder stage + DConv");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ch, IN_CH, ENC_KERNEL]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch, ch, 3]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch, ch, 1]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
    ];
    (def, t1, bindings)
}

#[test]
fn test_htdemucs_enc_stage_dconv_gelu_crown_after_fix() {
    let (def, t1, bindings) = build_enc_stage_with_dconv();
    let ch = ENC1_CH;
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(ibp_output.lower_upper().0.shape(), &[ch, t1]);
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    eprintln!(
        "HTDemucs encoder stage (Conv+GeLU + DConv+GNorm+GeLU) IBP: \
         [{ibp_lo}, {ibp_hi}], width={:.4}",
        ibp_hi - ibp_lo
    );

    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(crown_output.lower_upper().0.shape(), &[ch, t1]);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!(
        "HTDemucs encoder stage CROWN: method={method:?}, \
         [{crown_lo}, {crown_hi}], width={:.4}",
        crown_hi - crown_lo
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }
}

// ===========================================================================
// 5. Recording tests
// ===========================================================================

#[test]
fn test_htdemucs_two_stage_enc_gelu_record_crown_reverify() {
    let (def, bindings) = build_two_stage_encoder();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "htdemucs_two_stage_enc_gelu_crown_reverify",
    );
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "RECORD htdemucs_two_stage_enc_gelu_crown_reverify: [{lo}, {hi}], width={:.4}, \
         method={:?}, soundness={:?}",
        hi - lo,
        result.verification.method,
        result.verification.soundness_mode
    );
}

#[test]
fn test_htdemucs_multi_dconv_gelu_record_crown_reverify() {
    let (def, t, bindings) = build_multi_dconv_gelu();
    let input = uniform_bounds(&[ENC1_CH, t], 1.0);
    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_multi_dconv_gelu_crown_reverify",
        &conservative_config(),
    );
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "RECORD htdemucs_multi_dconv_gelu_crown_reverify: [{lo}, {hi}], width={:.4}, \
         method={:?}, soundness={:?}",
        hi - lo,
        result.verification.method,
        result.verification.soundness_mode
    );
}

#[test]
fn test_htdemucs_xformer_ffn_gelu_erf_record_crown_reverify() {
    let (def, bindings) = build_transformer_ffn_gelu_erf();
    let input = uniform_bounds(&[T_SEQ, D_MODEL], 1.0);
    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_xformer_ffn_gelu_erf_crown_reverify",
        &conservative_config(),
    );
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "RECORD htdemucs_xformer_ffn_gelu_erf_crown_reverify: [{lo}, {hi}], width={:.4}, \
         method={:?}, soundness={:?}",
        hi - lo,
        result.verification.method,
        result.verification.soundness_mode
    );
}

#[test]
fn test_htdemucs_enc_stage_dconv_gelu_record_crown_reverify() {
    let (def, _, bindings) = build_enc_stage_with_dconv();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);
    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_enc_stage_dconv_gelu_crown_reverify",
        &conservative_config(),
    );
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "RECORD htdemucs_enc_stage_dconv_gelu_crown_reverify: [{lo}, {hi}], width={:.4}, \
         method={:?}, soundness={:?}",
        hi - lo,
        result.verification.method,
        result.verification.soundness_mode
    );
}
