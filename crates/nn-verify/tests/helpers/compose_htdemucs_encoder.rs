// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HTDemucs encoder compose verification tests (IBP).
//!
//! Tests NY IBP bounds propagation through key encoder sub-stages
//! of the HTDemucs music source separation model. Each test uses small
//! symbolic dimensions for fast verification while preserving the
//! structural topology of the production architecture.
//!
//! 1. **Temporal Conv1d stride downsample**: Conv1d(k=8, s=4, p=2) + GELU.
//!    Verifies that stride downsampling preserves bounded outputs.
//!
//! 2. **Conv1d -> GroupNorm(G=1) chain**: Verifies GroupNorm normalisation
//!    bounds through the encoder path using Conservative NormBoundsMode
//!    for Sound classification.
//!
//! 3. **Isolated GELU activation bounds**: Confirms GELU IBP is tight
//!    (output in ~[-0.17, inf) for bounded inputs).
//!
//! 4. **Single DConv residual sublayer**: Conv1d(dilated) -> GroupNorm ->
//!    GELU -> Conv1d(1x1) -> GroupNorm -> GLU -> LayerScale -> residual.
//!    Tests the core DConv building block with Conservative Sound.
//!
//! 5. **Two-stage encoder stacking**: Encoder block 0 output feeds into
//!    encoder block 1 with doubled channels. Tests bounds propagation
//!    through sequential downsampling.
//!
//! 6. **Spectral encoder GroupNorm path**: 2D-shaped spectral input
//!    through Conv1d -> GroupNorm -> GELU on frequency-folded representation.
//!
//! All tests use Conservative NormBoundsMode to target Sound classification.
//! Part of #4186.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    uniform_bounds, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::{
    tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerificationSoundnessMode,
    VerifyConfig,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Input audio channels (matches HTDemucs stereo input).
const IN_CH: usize = 4;

/// First encoder stage output channels.
const ENC_CH: usize = 8;

/// Second encoder stage output channels (growth factor 2).
const ENC2_CH: usize = 16;

/// Temporal input length (small but valid for stride-4 Conv1d).
const T_IN: usize = 32;

/// Encoder Conv1d kernel size (production HTDemucs = 8).
const ENC_KERNEL: usize = 8;

/// Encoder Conv1d stride (production HTDemucs = 4).
const ENC_STRIDE: usize = 4;

/// Encoder Conv1d padding (production HTDemucs = kernel/4 = 2).
const ENC_PADDING: usize = 2;

/// DConv dilated convolution kernel size (production = 3).
const DCONV_KERNEL: usize = 3;

/// DConv compression ratio (production = 4).
const DCONV_COMPRESS: usize = 4;

/// Spectral input channels (stereo * real+imag = 4).
const SPEC_IN_CH: usize = 4;

/// Spectral first encoder output channels.
const SPEC_ENC_CH: usize = 8;

/// Small weight magnitude for stable IBP propagation.
const WEIGHT_MAG: f32 = 0.01;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

fn enc_t_out() -> usize {
    conv1d_out_len(T_IN, ENC_KERNEL, ENC_STRIDE, ENC_PADDING)
}

// ---------------------------------------------------------------------------
// DConv sub-layer builder (reusable across multiple tests)
// ---------------------------------------------------------------------------

/// Build a single DConv sub-layer inline.
///
/// Conv1d(dilated) -> GroupNorm(G=1) -> GELU -> Conv1d(1x1) -> GroupNorm(G=1) ->
/// GLU -> LayerScale -> residual_add.
fn add_dconv_sublayer(
    b: &mut TensorBlockBuilder,
    x: TensorNodeId,
    prefix: &str,
    depth_idx: usize,
    ch: usize,
    compressed: usize,
    t: usize,
    bindings: &mut Vec<TensorParamBinding>,
) -> TensorNodeId {
    let doubled = ch * 2;
    let dilation: usize = 1 << depth_idx;
    let dc_padding = dilation * (DCONV_KERNEL - 1) / 2;

    let cw = b.add_input(
        &format!("{prefix}_dc{depth_idx}_cw"),
        &[compressed, ch, DCONV_KERNEL],
    );
    let cb = b.add_input(&format!("{prefix}_dc{depth_idx}_cb"), &[compressed]);
    let ng = b.add_input(&format!("{prefix}_dc{depth_idx}_ng"), &[compressed]);
    let nb = b.add_input(&format!("{prefix}_dc{depth_idx}_nb"), &[compressed]);
    let ew = b.add_input(
        &format!("{prefix}_dc{depth_idx}_ew"),
        &[doubled, compressed, 1],
    );
    let eb = b.add_input(&format!("{prefix}_dc{depth_idx}_eb"), &[doubled]);
    let eng = b.add_input(&format!("{prefix}_dc{depth_idx}_eng"), &[doubled]);
    let enb = b.add_input(&format!("{prefix}_dc{depth_idx}_enb"), &[doubled]);
    let ls = b.add_input(&format!("{prefix}_dc{depth_idx}_ls"), &[ch]);
    let eps1 = b.add_input(&format!("{prefix}_dc{depth_idx}_eps1"), &[1]);
    let eps2 = b.add_input(&format!("{prefix}_dc{depth_idx}_eps2"), &[1]);

    // Dilated Conv1d
    let c1 = b.add_conv1d_full(
        x,
        cw,
        Some(cb),
        1,
        dc_padding,
        dilation,
        1,
        &[compressed, t],
    );
    let n1 = b.add_group_norm_g1(c1, eps1, Some(ng), Some(nb), compressed, t);
    let g1 = b.add_gelu(n1, &[compressed, t]);
    let c2 = b.add_conv1d(g1, ew, Some(eb), 1, 0, &[doubled, t]);
    let n2 = b.add_group_norm_g1(c2, eps2, Some(eng), Some(enb), doubled, t);
    let glu = b.add_glu(n2, 0, &[doubled, t]).expect("even dim");
    let ls_out = b.add_layer_scale(glu, ls, &[ch, t]);
    let out = b.add_binary_add(x, ls_out, &[ch, t]);

    // Bindings for this sub-layer (11 params)
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed, ch, DCONV_KERNEL]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled, compressed, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ch]),
        0.1f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    out
}

// ===========================================================================
// 1. Temporal Conv1d stride downsample + GELU bounds
// ===========================================================================

/// Build Conv1d(k=8, s=4, p=2) + GELU — the temporal encoder front-end.
fn build_conv1d_stride_gelu() -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    usize,
    Vec<TensorParamBinding>,
) {
    let t_out = enc_t_out();
    let mut b = TensorBlockBuilder::new("htdemucs_enc_conv1d_stride_gelu");
    let data = b.add_input("data", &[IN_CH, T_IN]);
    let conv_w = b.add_input("conv_w", &[ENC_CH, IN_CH, ENC_KERNEL]);
    let conv_b = b.add_input("conv_b", &[ENC_CH]);

    let conv_out = b.add_conv1d(
        data,
        conv_w,
        Some(conv_b),
        ENC_STRIDE,
        ENC_PADDING,
        &[ENC_CH, t_out],
    );
    let output = b.add_gelu(conv_out, &[ENC_CH, t_out]);

    let def = b.build(output).expect("valid conv1d stride + gelu");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ENC_CH, IN_CH, ENC_KERNEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_CH]), 0.0f32)),
    ];

    (def, t_out, bindings)
}

#[test]
fn test_htdemucs_enc_conv1d_stride_gelu_def_validates() {
    let (def, _, _) = build_conv1d_stride_gelu();
    def.validate()
        .expect("conv1d stride + gelu should validate");
}

#[test]
fn test_htdemucs_enc_conv1d_stride_gelu_ibp() {
    let (def, t_out, bindings) = build_conv1d_stride_gelu();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through conv1d stride + gelu");
    assert_eq!(output.lower_upper().0.shape(), &[ENC_CH, t_out]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs enc conv1d stride + GELU IBP: [{lo}, {hi}]");
    // GELU clamps lower to ~-0.17; with small weights bounds should be moderate
    assert!(lo.abs() < 100.0, "lower bound < 100, got {lo}");
    assert!(hi.abs() < 100.0, "upper bound < 100, got {hi}");
}

#[test]
fn test_htdemucs_enc_conv1d_stride_gelu_conservative_sound() {
    let (def, _, bindings) = build_conv1d_stride_gelu();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_enc_conv1d_stride_gelu",
        &conservative_config(),
    );

    // No normalization layers here, so Sound is expected even with Conservative
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conv1d + GELU (no norms) should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs enc conv1d stride + GELU (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 2. Conv1d -> GroupNorm(G=1) chain bounds
// ===========================================================================

/// Build Conv1d(stride) + GroupNorm(G=1) — encoder front-end with normalization.
fn build_conv1d_group_norm() -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    usize,
    Vec<TensorParamBinding>,
) {
    let t_out = enc_t_out();
    let mut b = TensorBlockBuilder::new("htdemucs_enc_conv1d_gnorm");
    let data = b.add_input("data", &[IN_CH, T_IN]);
    let conv_w = b.add_input("conv_w", &[ENC_CH, IN_CH, ENC_KERNEL]);
    let conv_b = b.add_input("conv_b", &[ENC_CH]);
    let gn_eps = b.add_input("gn_eps", &[1]);
    let gn_gamma = b.add_input("gn_gamma", &[ENC_CH]);
    let gn_beta = b.add_input("gn_beta", &[ENC_CH]);

    let conv_out = b.add_conv1d(
        data,
        conv_w,
        Some(conv_b),
        ENC_STRIDE,
        ENC_PADDING,
        &[ENC_CH, t_out],
    );
    let output = b.add_group_norm_g1(
        conv_out,
        gn_eps,
        Some(gn_gamma),
        Some(gn_beta),
        ENC_CH,
        t_out,
    );

    let def = b.build(output).expect("valid conv1d + group norm");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ENC_CH, IN_CH, ENC_KERNEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_CH]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_CH]), 0.0f32)),
    ];

    (def, t_out, bindings)
}

#[test]
fn test_htdemucs_enc_conv1d_gnorm_def_validates() {
    let (def, _, _) = build_conv1d_group_norm();
    def.validate().expect("conv1d + group norm should validate");
}

#[test]
fn test_htdemucs_enc_conv1d_gnorm_ibp() {
    let (def, t_out, bindings) = build_conv1d_group_norm();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through conv1d + group norm");
    assert_eq!(output.lower_upper().0.shape(), &[ENC_CH, t_out]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs enc conv1d + GroupNorm IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower bound finite, got {lo}");
    assert!(hi.is_finite(), "upper bound finite, got {hi}");
}

#[test]
fn test_htdemucs_enc_conv1d_gnorm_conservative_sound() {
    let (def, _, bindings) = build_conv1d_group_norm();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_enc_conv1d_gnorm",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative GroupNorm should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs enc conv1d + GroupNorm (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 3. Isolated GELU activation bounds
// ===========================================================================

/// Build isolated GELU on a small tensor — confirms GELU IBP tightness.
fn build_gelu_isolated() -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    let ch = ENC_CH;
    let t = 8;
    let mut b = TensorBlockBuilder::new("htdemucs_enc_gelu_isolated");
    let data = b.add_input("data", &[ch, t]);
    let output = b.add_gelu(data, &[ch, t]);

    let def = b.build(output).expect("valid gelu isolated");
    let bindings = vec![TensorParamBinding::Variable];

    (def, bindings)
}

#[test]
fn test_htdemucs_enc_gelu_isolated_def_validates() {
    let (def, _) = build_gelu_isolated();
    def.validate().expect("gelu isolated should validate");
}

#[test]
fn test_htdemucs_enc_gelu_isolated_ibp() {
    let (def, bindings) = build_gelu_isolated();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_CH, 8], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through gelu isolated");
    assert_eq!(output.lower_upper().0.shape(), &[ENC_CH, 8]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs GELU isolated IBP: [{lo}, {hi}]");
    // GELU(-1) ~ -0.159, GELU(1) ~ 0.841
    // IBP may be slightly wider but should respect activation bounds
    assert!(lo >= -0.5, "GELU lower should be >= -0.5, got {lo}");
    assert!(hi <= 1.5, "GELU upper should be <= 1.5, got {hi}");
}

#[test]
fn test_htdemucs_enc_gelu_isolated_crown() {
    let (def, bindings) = build_gelu_isolated();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_CH, 8], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[ENC_CH, 8]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs GELU isolated: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_htdemucs_enc_gelu_isolated_conservative_sound() {
    let (def, bindings) = build_gelu_isolated();
    let input = uniform_bounds(&[ENC_CH, 8], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_enc_gelu_isolated",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "GELU (no norms) should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs GELU isolated (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 4. Single DConv residual sublayer bounds
// ===========================================================================

/// Build a single DConv residual sublayer operating on encoder output dims.
fn build_single_dconv_sublayer() -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    let ch = ENC_CH;
    let compressed = ch / DCONV_COMPRESS;
    let t = enc_t_out();

    let mut b = TensorBlockBuilder::new("htdemucs_enc_single_dconv");
    let data = b.add_input("data", &[ch, t]);

    let mut bindings = vec![TensorParamBinding::Variable];
    let output = add_dconv_sublayer(&mut b, data, "enc", 0, ch, compressed, t, &mut bindings);

    let def = b.build(output).expect("valid single dconv sublayer");
    (def, bindings)
}

#[test]
fn test_htdemucs_enc_single_dconv_def_validates() {
    let (def, _) = build_single_dconv_sublayer();
    def.validate()
        .expect("single dconv sublayer should validate");
}

#[test]
fn test_htdemucs_enc_single_dconv_ibp() {
    let (def, bindings) = build_single_dconv_sublayer();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let t = enc_t_out();
    let input = uniform_bounds(&[ENC_CH, t], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through single dconv sublayer");
    assert_eq!(output.lower_upper().0.shape(), &[ENC_CH, t]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs single DConv IBP: [{lo}, {hi}]");
    // Residual connection bounds: input + LayerScale(0.1) * dconv_output
    // Should be close to input range
    assert!(lo.abs() < 1e4, "lower bound < 1e4, got {lo}");
    assert!(hi.abs() < 1e4, "upper bound < 1e4, got {hi}");
}

#[test]
fn test_htdemucs_enc_single_dconv_conservative_sound() {
    let (def, bindings) = build_single_dconv_sublayer();
    let t = enc_t_out();
    let input = uniform_bounds(&[ENC_CH, t], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_enc_single_dconv",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative single DConv should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs single DConv (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 5. Two-stage encoder stacking bounds
// ===========================================================================

/// Build two encoder stages: block 0 (IN_CH -> ENC_CH) followed by
/// block 1 (ENC_CH -> ENC2_CH). Each block: Conv1d(stride) + GELU + DConv.
fn build_two_stage_encoder() -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    usize,
    Vec<TensorParamBinding>,
) {
    let t1 = enc_t_out();
    let t2 = conv1d_out_len(t1, ENC_KERNEL, ENC_STRIDE, ENC_PADDING);
    let compressed1 = ENC_CH / DCONV_COMPRESS;
    let compressed2 = ENC2_CH / DCONV_COMPRESS;

    let mut b = TensorBlockBuilder::new("htdemucs_enc_two_stage");

    // Stage 0: Conv1d(IN_CH -> ENC_CH) + GELU + 1 DConv
    let data = b.add_input("data", &[IN_CH, T_IN]);
    let conv0_w = b.add_input("conv0_w", &[ENC_CH, IN_CH, ENC_KERNEL]);
    let conv0_b = b.add_input("conv0_b", &[ENC_CH]);

    let conv0_out = b.add_conv1d(
        data,
        conv0_w,
        Some(conv0_b),
        ENC_STRIDE,
        ENC_PADDING,
        &[ENC_CH, t1],
    );
    let x0 = b.add_gelu(conv0_out, &[ENC_CH, t1]);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ENC_CH, IN_CH, ENC_KERNEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_CH]), 0.0f32)),
    ];

    let x0_dconv = add_dconv_sublayer(&mut b, x0, "s0", 0, ENC_CH, compressed1, t1, &mut bindings);

    // GLU rewrite for stage 0
    let doubled0 = ENC_CH * 2;
    let rw0_w = b.add_input("rw0_w", &[doubled0, ENC_CH, 1]);
    let rw0_b = b.add_input("rw0_b", &[doubled0]);
    let rw0_out = b.add_conv1d(x0_dconv, rw0_w, Some(rw0_b), 1, 0, &[doubled0, t1]);
    let stage0_out = b.add_glu(rw0_out, 0, &[doubled0, t1]).expect("even dim");

    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled0, ENC_CH, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled0]),
        0.0f32,
    )));

    // Stage 1: Conv1d(ENC_CH -> ENC2_CH) + GELU + 1 DConv
    let conv1_w = b.add_input("conv1_w", &[ENC2_CH, ENC_CH, ENC_KERNEL]);
    let conv1_b = b.add_input("conv1_b", &[ENC2_CH]);
    let conv1_out = b.add_conv1d(
        stage0_out,
        conv1_w,
        Some(conv1_b),
        ENC_STRIDE,
        ENC_PADDING,
        &[ENC2_CH, t2],
    );
    let x1 = b.add_gelu(conv1_out, &[ENC2_CH, t2]);

    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ENC2_CH, ENC_CH, ENC_KERNEL]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ENC2_CH]),
        0.0f32,
    )));

    let output = add_dconv_sublayer(&mut b, x1, "s1", 0, ENC2_CH, compressed2, t2, &mut bindings);

    let def = b.build(output).expect("valid two-stage encoder");
    (def, t2, bindings)
}

#[test]
fn test_htdemucs_enc_two_stage_def_validates() {
    let (def, _, _) = build_two_stage_encoder();
    def.validate().expect("two-stage encoder should validate");
}

#[test]
fn test_htdemucs_enc_two_stage_ibp() {
    let (def, t2, bindings) = build_two_stage_encoder();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through two-stage encoder");
    assert_eq!(output.lower_upper().0.shape(), &[ENC2_CH, t2]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs two-stage encoder IBP: [{lo}, {hi}]");
    // Two stages of Conv1d(stride) + DConv; bounds may grow but should stay finite
    assert!(lo.is_finite(), "lower bound finite, got {lo}");
    assert!(hi.is_finite(), "upper bound finite, got {hi}");
    assert!(lo.abs() < 1e8, "lower bound < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "upper bound < 1e8, got {hi}");
}

#[test]
fn test_htdemucs_enc_two_stage_conservative_sound() {
    let (def, _, bindings) = build_two_stage_encoder();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_enc_two_stage",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative two-stage encoder should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs two-stage encoder (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

#[test]
fn test_htdemucs_enc_two_stage_crown() {
    let (def, t2, bindings) = build_two_stage_encoder();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[ENC2_CH, t2]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs two-stage encoder: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 6. Spectral encoder GroupNorm path bounds
// ===========================================================================

/// Build spectral encoder front-end: Conv1d(stride) + GroupNorm(G=1) + GELU.
/// Spectral path operates on frequency-folded [C, F] representation.
fn build_spectral_gnorm_gelu() -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    usize,
    Vec<TensorParamBinding>,
) {
    let f_in: usize = 16; // frequency bins (small symbolic dim)
    let f_out = conv1d_out_len(f_in, ENC_KERNEL, ENC_STRIDE, ENC_PADDING);

    let mut b = TensorBlockBuilder::new("htdemucs_enc_spectral_gnorm_gelu");
    let data = b.add_input("data", &[SPEC_IN_CH, f_in]);
    let conv_w = b.add_input("conv_w", &[SPEC_ENC_CH, SPEC_IN_CH, ENC_KERNEL]);
    let conv_b = b.add_input("conv_b", &[SPEC_ENC_CH]);
    let gn_eps = b.add_input("gn_eps", &[1]);
    let gn_gamma = b.add_input("gn_gamma", &[SPEC_ENC_CH]);
    let gn_beta = b.add_input("gn_beta", &[SPEC_ENC_CH]);

    let conv_out = b.add_conv1d(
        data,
        conv_w,
        Some(conv_b),
        ENC_STRIDE,
        ENC_PADDING,
        &[SPEC_ENC_CH, f_out],
    );
    let normed = b.add_group_norm_g1(
        conv_out,
        gn_eps,
        Some(gn_gamma),
        Some(gn_beta),
        SPEC_ENC_CH,
        f_out,
    );
    let output = b.add_gelu(normed, &[SPEC_ENC_CH, f_out]);

    let def = b.build(output).expect("valid spectral gnorm gelu");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[SPEC_ENC_CH, SPEC_IN_CH, ENC_KERNEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SPEC_ENC_CH]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SPEC_ENC_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SPEC_ENC_CH]), 0.0f32)),
    ];

    (def, f_out, bindings)
}

#[test]
fn test_htdemucs_enc_spectral_gnorm_gelu_def_validates() {
    let (def, _, _) = build_spectral_gnorm_gelu();
    def.validate().expect("spectral gnorm gelu should validate");
}

#[test]
fn test_htdemucs_enc_spectral_gnorm_gelu_ibp() {
    let (def, f_out, bindings) = build_spectral_gnorm_gelu();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SPEC_IN_CH, 16], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through spectral gnorm gelu");
    assert_eq!(output.lower_upper().0.shape(), &[SPEC_ENC_CH, f_out]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs spectral encoder GroupNorm + GELU IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower bound finite, got {lo}");
    assert!(hi.is_finite(), "upper bound finite, got {hi}");
}

#[test]
fn test_htdemucs_enc_spectral_gnorm_gelu_conservative_sound() {
    let (def, _, bindings) = build_spectral_gnorm_gelu();
    let input = uniform_bounds(&[SPEC_IN_CH, 16], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_enc_spectral_gnorm_gelu",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative spectral GroupNorm + GELU should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs spectral GroupNorm + GELU (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

#[test]
fn test_htdemucs_enc_spectral_gnorm_gelu_crown() {
    let (def, f_out, bindings) = build_spectral_gnorm_gelu();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SPEC_IN_CH, 16], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SPEC_ENC_CH, f_out]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs spectral GroupNorm + GELU: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}
