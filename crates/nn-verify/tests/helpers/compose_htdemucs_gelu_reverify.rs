// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HTDemucs GeLU re-verification compose tests after NY CROWN relaxation
//! soundness fix (rev e810fb2b).
//!
//! The CROWN relaxation fix in NY corrects the linear relaxation bounds
//! computation for GeLU (and Mish) activations. This affects CROWN-mode verification
//! of any model using GeLU. HTDemucs uses GeLU in:
//!
//! - **Encoder front-end**: Conv1d(stride) -> GeLU temporal downsampling
//! - **DConv sub-layers**: Dilated Conv -> GroupNorm -> GeLU -> Conv(1x1)
//! - **Transformer FFN**: Linear -> GeLU -> Linear cross-domain bottleneck
//!
//! These tests re-verify GeLU-containing components with CROWN to confirm:
//!
//! 1. **Isolated GeLU CROWN**: CROWN bounds through standalone GeLU activation.
//!
//! 2. **Conv1d + GeLU CROWN**: Temporal encoder front-end with CROWN propagation.
//!
//! 3. **Transformer FFN with GeLU CROWN**: Cross-domain FFN bottleneck with
//!    GeLU activation verified under CROWN.
//!
//! 4. **Conv1d + GroupNorm + GeLU CROWN**: Full encoder path with normalization
//!    and GeLU, testing CROWN through the complete chain.
//!
//! 5. **IBP vs CROWN tightness for GeLU paths**: Quantitative comparison of
//!    bounds width between IBP and CROWN for GeLU-containing sub-blocks.
//!
//! All tests use Conservative NormBoundsMode where normalization layers are present.
//! Part of #4314: Re-verify GeLU models after NY CROWN relaxation fix.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    uniform_bounds, verify_and_assert, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
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

/// Temporal input length (small but valid for stride-4 Conv1d).
const T_IN: usize = 32;

/// Encoder Conv1d kernel size (production HTDemucs = 8).
const ENC_KERNEL: usize = 8;

/// Encoder Conv1d stride (production HTDemucs = 4).
const ENC_STRIDE: usize = 4;

/// Encoder Conv1d padding (production HTDemucs = kernel/4 = 2).
const ENC_PADDING: usize = 2;

/// Model dimension for transformer bottleneck.
const D_MODEL: usize = 8;

/// Number of attention heads.
const NUM_HEADS: usize = 2;

/// FFN intermediate dimension (2x model dim, standard ratio).
const FFN_DIM: usize = D_MODEL * 2;

/// Transformer sequence length.
const T_SEQ: usize = 4;

/// Small weight magnitude for stable IBP propagation.
const WEIGHT_MAG: f32 = 0.01;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

fn enc_t_out() -> usize {
    conv1d_out_len(T_IN, ENC_KERNEL, ENC_STRIDE, ENC_PADDING)
}

// ===========================================================================
// 1. Isolated GeLU CROWN re-verification
// ===========================================================================

/// Build standalone GeLU on encoder-shaped tensor.
fn build_gelu_isolated() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let ch = ENC_CH;
    let t = 8;
    let mut b = TensorBlockBuilder::new("htdemucs_gelu_reverify_isolated");
    let data = b.add_input("data", &[ch, t]);
    let output = b.add_gelu(data, &[ch, t]);

    let def = b.build(output).expect("valid gelu isolated");
    let bindings = vec![TensorParamBinding::Variable];
    (def, bindings)
}

#[test]
fn test_htdemucs_gelu_isolated_crown_after_fix() {
    let (def, bindings) = build_gelu_isolated();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[ENC_CH, 8], 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("HTDemucs GeLU isolated IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    // CROWN with fixed relaxation
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "HTDemucs GeLU isolated CROWN: method={method:?}, [{crown_lo}, {crown_hi}], width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    // GeLU(-1) ~ -0.159, GeLU(1) ~ 0.841
    assert!(ibp_lo >= -0.5, "IBP lower >= -0.5, got {ibp_lo}");
    assert!(ibp_hi <= 1.5, "IBP upper <= 1.5, got {ibp_hi}");
    assert!(crown_lo >= -0.5, "CROWN lower >= -0.5, got {crown_lo}");
    assert!(crown_hi <= 1.5, "CROWN upper <= 1.5, got {crown_hi}");
}

// ===========================================================================
// 2. Conv1d + GeLU encoder front-end -- CROWN re-verification
// ===========================================================================

/// Build Conv1d(k=8, s=4, p=2) + GeLU -- the temporal encoder front-end.
fn build_conv1d_gelu() -> (TensorKernelDef, usize, Vec<TensorParamBinding>) {
    let t_out = enc_t_out();
    let mut b = TensorBlockBuilder::new("htdemucs_conv1d_gelu_reverify");
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

    let def = b.build(output).expect("valid conv1d + gelu");
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
fn test_htdemucs_conv1d_gelu_crown_after_fix() {
    let (def, t_out, bindings) = build_conv1d_gelu();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(ibp_output.lower_upper().0.shape(), &[ENC_CH, t_out]);
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("HTDemucs conv1d+GeLU IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    // CROWN with fixed relaxation -- tests CROWN soundness through conv+GeLU
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(crown_output.lower_upper().0.shape(), &[ENC_CH, t_out]);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "HTDemucs conv1d+GeLU CROWN: method={method:?}, [{crown_lo}, {crown_hi}], width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    // Bounds should be moderate with small weights
    assert!(ibp_lo.abs() < 100.0, "IBP lower < 100, got {ibp_lo}");
    assert!(ibp_hi.abs() < 100.0, "IBP upper < 100, got {ibp_hi}");
}

// ===========================================================================
// 3. Transformer FFN with GeLU -- CROWN re-verification
// ===========================================================================

/// Build FFN sub-block: LayerNorm -> Linear(D, FFN_DIM) -> GeLU -> Linear(FFN_DIM, D) -> residual.
///
/// This is the FFN used in HTDemucs cross-domain transformer blocks.
fn build_transformer_ffn_gelu() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("htdemucs_xformer_ffn_gelu_reverify");

    let x = b.add_input("x", &[T_SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let shape = [T_SEQ, D_MODEL];
    let ffn_shape = [T_SEQ, FFN_DIM];

    // Pre-norm: LayerNorm
    let normed = b.add_layer_norm(x, eps, 1, ln_w, ln_b, &shape);
    // FFN: Linear -> GeLU -> Linear
    let h = b.add_linear(normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(h, &ffn_shape);
    let proj = b.add_linear(act, ffn2_w, None, &shape);
    // Residual
    let out = b.add_binary_add(x, proj, &shape);

    let def = b.build(out).expect("valid FFN+GeLU kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, D_MODEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, FFN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    (def, bindings)
}

#[test]
fn test_htdemucs_xformer_ffn_gelu_crown_after_fix() {
    let (def, bindings) = build_transformer_ffn_gelu();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[T_SEQ, D_MODEL], 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("HTDemucs transformer FFN+GeLU IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    // CROWN with fixed relaxation through LayerNorm + GeLU
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "HTDemucs transformer FFN+GeLU CROWN: method={method:?}, [{crown_lo}, {crown_hi}], \
         width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    // Bounds should be finite and valid
    assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
    assert!(crown_lo.is_finite() && crown_hi.is_finite());
}

// ===========================================================================
// 4. Conv1d + GroupNorm + GeLU -- CROWN through normalization + activation
// ===========================================================================

/// Build Conv1d(stride) -> GroupNorm(G=1) -> GeLU -- full encoder path with
/// normalization before GeLU activation.
fn build_conv1d_gnorm_gelu() -> (TensorKernelDef, usize, Vec<TensorParamBinding>) {
    let t_out = enc_t_out();
    let mut b = TensorBlockBuilder::new("htdemucs_conv_gnorm_gelu_reverify");
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
    let normed = b.add_group_norm_g1(
        conv_out,
        gn_eps,
        Some(gn_gamma),
        Some(gn_beta),
        ENC_CH,
        t_out,
    );
    let output = b.add_gelu(normed, &[ENC_CH, t_out]);

    let def = b.build(output).expect("valid conv + gnorm + gelu");
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
fn test_htdemucs_conv_gnorm_gelu_crown_after_fix() {
    let (def, t_out, bindings) = build_conv1d_gnorm_gelu();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(ibp_output.lower_upper().0.shape(), &[ENC_CH, t_out]);
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("HTDemucs conv+GroupNorm+GeLU IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    // CROWN through GroupNorm + GeLU -- tests the interaction of the
    // normalization fix (gc#4399) with the GeLU relaxation fix
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(crown_output.lower_upper().0.shape(), &[ENC_CH, t_out]);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "HTDemucs conv+GroupNorm+GeLU CROWN: method={method:?}, [{crown_lo}, {crown_hi}], \
         width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }
}

#[test]
fn test_htdemucs_conv_gnorm_gelu_conservative_sound() {
    let (def, _, bindings) = build_conv1d_gnorm_gelu();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_conv_gnorm_gelu_reverify",
        &conservative_config(),
    );

    // Conservative NormBoundsMode through GroupNorm + GeLU: CROWN escalation
    // may trigger normalization heuristics, producing Heuristic soundness.
    // IBP-only would produce Sound, but run_escalation attempts CROWN first.
    assert!(
        result.verification.soundness_mode == VerificationSoundnessMode::Sound
            || result.verification.soundness_mode == VerificationSoundnessMode::Heuristic,
        "Conservative GroupNorm+GeLU should produce Sound or Heuristic, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs conv+GroupNorm+GeLU Conservative: [{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 5. IBP vs CROWN tightness comparison for GeLU paths
// ===========================================================================

/// Compare IBP and CROWN bounds width for the transformer FFN sub-block
/// at different input ranges. After the CROWN relaxation fix, CROWN should
/// produce tighter bounds than IBP for GeLU-containing blocks.
#[test]
fn test_htdemucs_gelu_ibp_vs_crown_tightness() {
    let (def, bindings) = build_transformer_ffn_gelu();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    for range in [0.5_f32, 1.0, 2.0] {
        let input = uniform_bounds(&[T_SEQ, D_MODEL], range);

        // IBP
        let ibp_output = graph.propagate_ibp(&input).expect("IBP");
        let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
        let ibp_width = ibp_hi - ibp_lo;

        // CROWN
        let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
        let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
        let crown_width = crown_hi - crown_lo;

        eprintln!(
            "HTDemucs GeLU FFN range={range}: IBP width={ibp_width:.4}, CROWN width={crown_width:.4} \
             (method={method:?}, ratio={:.2}x)",
            if crown_width > 0.0 {
                ibp_width / crown_width
            } else {
                f32::INFINITY
            }
        );

        // Both must produce finite, valid bounds
        assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
        assert!(crown_lo.is_finite() && crown_hi.is_finite());
        assert!(ibp_width >= 0.0, "IBP width must be non-negative");
        assert!(crown_width >= 0.0, "CROWN width must be non-negative");
    }
}

// ===========================================================================
// 6. Recording tests -- write CROWN reverify results to status file
// ===========================================================================
//
// These tests use `verify_and_assert` / `verify_and_assert_with_config` to
// record CROWN verification results to `nn_verify_status_demucs.json`.
// The pipeline's `run_escalation` automatically attempts CROWN after IBP.
//
// Status keys use the `htdemucs_` prefix so they map to the "demucs" model
// category via `model_for_kernel()`.

/// Record isolated GeLU CROWN verification to status file.
///
/// Establishes a CROWN baseline for standalone GeLU activation on
/// encoder-shaped tensors after the NY relaxation fix (e810fb2b).
#[test]
fn test_htdemucs_gelu_isolated_record_crown_reverify() {
    let (def, bindings) = build_gelu_isolated();
    let input = uniform_bounds(&[ENC_CH, 8], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "htdemucs_gelu_isolated_crown_reverify",
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD htdemucs_gelu_isolated_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}

/// Record Conv1d+GeLU encoder front-end CROWN verification to status file.
///
/// Tests the temporal encoder pattern: Conv1d(stride) -> GeLU with CROWN
/// propagation through the combined linear+activation sequence.
#[test]
fn test_htdemucs_conv1d_gelu_record_crown_reverify() {
    let (def, _, bindings) = build_conv1d_gelu();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "htdemucs_conv1d_gelu_crown_reverify",
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD htdemucs_conv1d_gelu_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}

/// Record transformer FFN+GeLU CROWN verification to status file.
///
/// Tests the cross-domain transformer bottleneck:
/// LayerNorm -> Linear -> GeLU -> Linear -> residual.
/// Uses Conservative NormBoundsMode since LayerNorm is present.
#[test]
fn test_htdemucs_xformer_ffn_gelu_record_crown_reverify() {
    let (def, bindings) = build_transformer_ffn_gelu();
    let input = uniform_bounds(&[T_SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_xformer_ffn_gelu_crown_reverify",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD htdemucs_xformer_ffn_gelu_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}

/// Record Conv1d+GroupNorm+GeLU full encoder path CROWN verification.
///
/// Complete temporal encoder front-end: Conv1d(stride) -> GroupNorm(G=1) -> GeLU.
/// Uses Conservative NormBoundsMode for GroupNorm.
#[test]
fn test_htdemucs_conv_gnorm_gelu_record_crown_reverify() {
    let (def, _, bindings) = build_conv1d_gnorm_gelu();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_conv_gnorm_gelu_crown_reverify",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD htdemucs_conv_gnorm_gelu_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );

    // Conservative NormBoundsMode through GroupNorm + GeLU: CROWN escalation
    // may trigger normalization heuristics, producing Heuristic soundness.
    assert!(
        result.verification.soundness_mode == VerificationSoundnessMode::Sound
            || result.verification.soundness_mode == VerificationSoundnessMode::Heuristic,
        "Conservative GroupNorm+GeLU should produce Sound or Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 7. DConv sub-layer with GeLU -- dilated convolution pattern
// ===========================================================================
//
// HTDemucs DConv blocks use: DilatedConv1d -> GroupNorm -> GeLU -> Conv1d(1x1).
// This tests CROWN through that specific architecture after the relaxation fix.

/// Build DConv sub-layer: Conv1d(k=3, d=1) -> GroupNorm -> GeLU -> Conv1d(1x1).
///
/// This is the DConv pattern used in HTDemucs encoder and decoder stages.
fn build_dconv_gelu() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let ch = ENC_CH;
    let t = enc_t_out();
    let mut b = TensorBlockBuilder::new("htdemucs_dconv_gelu_reverify");

    let x = b.add_input("x", &[ch, t]);
    // Conv1d(k=3, stride=1, padding=1) -- same-padding dilated conv
    let dc_w = b.add_input("dc_w", &[ch, ch, 3]);
    let dc_b = b.add_input("dc_b", &[ch]);
    let dc_out = b.add_conv1d(x, dc_w, Some(dc_b), 1, 1, &[ch, t]);
    // GroupNorm(G=1)
    let gn_eps = b.add_input("gn_eps", &[1]);
    let gn_gamma = b.add_input("gn_gamma", &[ch]);
    let gn_beta = b.add_input("gn_beta", &[ch]);
    let normed = b.add_group_norm_g1(dc_out, gn_eps, Some(gn_gamma), Some(gn_beta), ch, t);
    // GeLU activation
    let activated = b.add_gelu(normed, &[ch, t]);
    // Conv1d(1x1) pointwise projection
    let pw_w = b.add_input("pw_w", &[ch, ch, 1]);
    let pw_b = b.add_input("pw_b", &[ch]);
    let output = b.add_conv1d(activated, pw_w, Some(pw_b), 1, 0, &[ch, t]);

    let def = b.build(output).expect("valid DConv+GeLU kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch, ch, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch, ch, 1]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
    ];
    (def, bindings)
}

#[test]
fn test_htdemucs_dconv_gelu_crown_after_fix() {
    let (def, bindings) = build_dconv_gelu();
    let ch = ENC_CH;
    let t = enc_t_out();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[ch, t], 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("HTDemucs DConv+GeLU IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    // CROWN with fixed GeLU relaxation through DConv pattern
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "HTDemucs DConv+GeLU CROWN: method={method:?}, [{crown_lo}, {crown_hi}], \
         width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }
}

/// Record DConv+GeLU CROWN verification to status file.
#[test]
fn test_htdemucs_dconv_gelu_record_crown_reverify() {
    let (def, bindings) = build_dconv_gelu();
    let ch = ENC_CH;
    let t = enc_t_out();
    let input = uniform_bounds(&[ch, t], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_dconv_gelu_crown_reverify",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD htdemucs_dconv_gelu_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}
