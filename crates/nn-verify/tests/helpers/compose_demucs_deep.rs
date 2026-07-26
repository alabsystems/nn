// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep Demucs compose verification tests targeting vacuous entry promotion.
//!
//! The 7 vacuous entries in `nn_verify_status_demucs.json` all use heuristic
//! soundness mode (ForwardMode normalization approximation) which produces
//! vacuously wide bounds. Re-verifying with `NormBoundsMode::Conservative`
//! produces Sound soundness — provably sound IBP through normalization layers
//! without heuristic linearization.
//!
//! Targets:
//!   1. `demucs_spectral_encoder_block` — vacuous → sound
//!   2. `demucs_spectral_encoder_prod_dconv` — vacuous → sound
//!   3. `demucs_spectral_decoder_dconv` — vacuous → sound
//!   4. `demucs_temporal_encoder_block` — vacuous → sound
//!   5. `demucs_temporal_encoder_prod_block0` — vacuous → sound
//!   6. `demucs_temporal_decoder_block` — vacuous → sound
//!   7. `demucs_cross_domain_bottleneck` — vacuous → sound
//!
//! Part of verification gap closure for dvoice production models.

// Re-import `common` module so child `#[path]` helpers can resolve `super::common`.
#[allow(unused_imports)]
use super::common;

// Re-use existing builder helpers from other compose test families.
#[path = "spectral_encoder.rs"]
mod spectral_enc_helpers;

#[path = "temporal_encoder.rs"]
mod temporal_enc_helpers;

use common::{assert_bounds_valid, bounds_min_max, uniform_bounds, verify_and_assert_with_config};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{NormBoundsMode, TensorParamBinding, VerificationSoundnessMode, VerifyConfig};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Conservative config for Sound soundness via IBP-validated normalization
// ---------------------------------------------------------------------------

/// VerifyConfig using Conservative NormBoundsMode which avoids heuristic
/// normalization linearization entirely. This produces `Sound` classification
/// from NY because no heuristic switches are enabled.
fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ===========================================================================
// 1. Spectral encoder block — Sound re-verification
// ===========================================================================

/// Re-verify `demucs_spectral_encoder_block` with Conservative mode.
///
/// The existing entry uses heuristic ForwardMode normalization producing
/// vacuously wide bounds (~1.3e32). Conservative IBP is provably sound
/// and should produce finite bounds that record as Sound.
#[test]
fn test_sound_spectral_encoder_block() {
    let (def, conv_f_out, _) = spectral_enc_helpers::build_spectral_encoder_block();
    let bindings = spectral_enc_helpers::spectral_encoder_block_bindings();
    let input = uniform_bounds(
        &[
            spectral_enc_helpers::IN_CHANNELS,
            spectral_enc_helpers::F_IN,
        ],
        1.0,
    );

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "demucs_spectral_encoder_block",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Spectral encoder block (Conservative): bounds=[{lo}, {hi}], width={}, soundness={:?}",
        result.verification.output_width, result.verification.soundness_mode
    );

    let (lo_arr, _) = result.output_bounds.lower_upper();
    assert_eq!(
        lo_arr.shape(),
        &[spectral_enc_helpers::OUT_CHANNELS, conv_f_out],
        "output shape mismatch"
    );
}

// ===========================================================================
// 2. Spectral encoder prod DConv — Sound re-verification
// ===========================================================================

/// Spectral DConv sub-layer dimensions (matching production DCONV_COMPRESS=4).
const SPEC_DCONV_CH: usize = 8;
const SPEC_DCONV_COMPRESSED: usize = 2;
const SPEC_DCONV_F: usize = 4;
const SPEC_DCONV_KERNEL: usize = 3;

fn build_spectral_dconv_block() -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    let ch = SPEC_DCONV_CH;
    let compressed = SPEC_DCONV_COMPRESSED;
    let doubled = ch * 2;
    let f = SPEC_DCONV_F;

    let mut b = TensorBlockBuilder::new("demucs_spec_dconv_verify");
    let data = b.add_input("data", &[ch, f]);

    // DConv sub-layer: Conv1d(dilated) -> GroupNorm(G=1) -> GELU ->
    // Conv1d(1x1) -> GroupNorm(G=1) -> GLU -> LayerScale -> residual_add
    let cw = b.add_input("dc_cw", &[compressed, ch, SPEC_DCONV_KERNEL]);
    let cb = b.add_input("dc_cb", &[compressed]);
    let ng = b.add_input("dc_ng", &[compressed]);
    let nb = b.add_input("dc_nb", &[compressed]);
    let ew = b.add_input("dc_ew", &[doubled, compressed, 1]);
    let eb = b.add_input("dc_eb", &[doubled]);
    let eng = b.add_input("dc_eng", &[doubled]);
    let enb = b.add_input("dc_enb", &[doubled]);
    let ls = b.add_input("dc_ls", &[ch]);
    let eps1 = b.add_input("dc_eps1", &[1]);
    let eps2 = b.add_input("dc_eps2", &[1]);

    let dc_padding = (SPEC_DCONV_KERNEL - 1) / 2; // dilation=1

    // Dilated Conv1d
    let c1 = b.add_conv1d_full(data, cw, Some(cb), 1, dc_padding, 1, 1, &[compressed, f]);
    // GroupNorm(G=1)
    let n1 = b.add_group_norm_g1(c1, eps1, Some(ng), Some(nb), compressed, f);
    // GELU
    let g1 = b.add_gelu(n1, &[compressed, f]);
    // Conv1d expand
    let c2 = b.add_conv1d(g1, ew, Some(eb), 1, 0, &[doubled, f]);
    // GroupNorm(G=1)
    let n2 = b.add_group_norm_g1(c2, eps2, Some(eng), Some(enb), doubled, f);
    // GLU
    let glu = b.add_glu(n2, 0, &[doubled, f]).expect("even dim");
    // LayerScale
    let ls_out = b.add_layer_scale(glu, ls, &[ch, f]);
    // Residual
    let output = b.add_binary_add(data, ls_out, &[ch, f]);

    let def = b.build(output).expect("valid spectral DConv block");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed, ch, SPEC_DCONV_KERNEL]),
        0.01f32,
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
        0.01f32,
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

    (def, bindings)
}

/// Re-verify `demucs_spectral_encoder_prod_dconv` with Conservative mode.
#[test]
fn test_sound_spectral_encoder_prod_dconv() {
    let (def, bindings) = build_spectral_dconv_block();
    let input = uniform_bounds(&[SPEC_DCONV_CH, SPEC_DCONV_F], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "demucs_spectral_encoder_prod_dconv",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Spectral encoder prod DConv (Conservative): bounds=[{lo}, {hi}], width={}, soundness={:?}",
        result.verification.output_width, result.verification.soundness_mode
    );
}

// ===========================================================================
// 3. Spectral decoder DConv — Sound re-verification
// ===========================================================================

/// Re-verify `demucs_spectral_decoder_dconv` with Conservative mode.
///
/// Uses the same DConv topology as the encoder (the DConv sub-layer is
/// symmetric between encoder and decoder in HTDemucs).
#[test]
fn test_sound_spectral_decoder_dconv() {
    let (def, bindings) = build_spectral_dconv_block();
    let input = uniform_bounds(&[SPEC_DCONV_CH, SPEC_DCONV_F], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "demucs_spectral_decoder_dconv",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Spectral decoder DConv (Conservative): bounds=[{lo}, {hi}], width={}, soundness={:?}",
        result.verification.output_width, result.verification.soundness_mode
    );
}

// ===========================================================================
// 4. Temporal encoder block — Sound re-verification
// ===========================================================================

/// Re-verify `demucs_temporal_encoder_block` with Conservative mode.
#[test]
fn test_sound_temporal_encoder_block() {
    let (def, conv_t_out, _) = temporal_enc_helpers::build_encoder_block();
    let bindings = temporal_enc_helpers::encoder_block_bindings();
    let input = uniform_bounds(
        &[
            temporal_enc_helpers::IN_CHANNELS,
            temporal_enc_helpers::T_IN,
        ],
        1.0,
    );

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "demucs_temporal_encoder_block",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Temporal encoder block (Conservative): bounds=[{lo}, {hi}], width={}, soundness={:?}",
        result.verification.output_width, result.verification.soundness_mode
    );

    let (lo_arr, _) = result.output_bounds.lower_upper();
    assert_eq!(
        lo_arr.shape(),
        &[temporal_enc_helpers::OUT_CHANNELS, conv_t_out],
        "output shape mismatch"
    );
}

// ===========================================================================
// 5. Temporal encoder prod block0 — Sound re-verification
// ===========================================================================

const TEMP_PROD_IN_CH: usize = 4;
const TEMP_PROD_OUT_CH: usize = 8;
const TEMP_PROD_T_IN: usize = 16;
const TEMP_PROD_CONV_KERNEL: usize = 8;
const TEMP_PROD_CONV_STRIDE: usize = 4;
const TEMP_PROD_CONV_PADDING: usize = 2;
const TEMP_PROD_DCONV_KERNEL: usize = 3;
const TEMP_PROD_DCONV_DEPTH: usize = 2;
const TEMP_PROD_DCONV_COMPRESS: usize = 4;

fn build_temporal_prod_block0() -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    usize,
    Vec<TensorParamBinding>,
) {
    use super::common::conv1d_out_len;

    let in_ch = TEMP_PROD_IN_CH;
    let out_ch = TEMP_PROD_OUT_CH;
    let compressed = out_ch / TEMP_PROD_DCONV_COMPRESS;
    let doubled = out_ch * 2;

    let mut b = TensorBlockBuilder::new("demucs_temp_prod_block0_verify");

    let data = b.add_input("data", &[in_ch, TEMP_PROD_T_IN]);
    let conv_weight = b.add_input("conv_w", &[out_ch, in_ch, TEMP_PROD_CONV_KERNEL]);
    let conv_bias = b.add_input("conv_b", &[out_ch]);

    let conv_t_out = conv1d_out_len(
        TEMP_PROD_T_IN,
        TEMP_PROD_CONV_KERNEL,
        TEMP_PROD_CONV_STRIDE,
        TEMP_PROD_CONV_PADDING,
    );

    // Conv1d stride downsample
    let conv_out = b.add_conv1d(
        data,
        conv_weight,
        Some(conv_bias),
        TEMP_PROD_CONV_STRIDE,
        TEMP_PROD_CONV_PADDING,
        &[out_ch, conv_t_out],
    );
    let gelu_out = b.add_gelu(conv_out, &[out_ch, conv_t_out]);

    // DConv sub-layers
    let mut dconv_out = gelu_out;

    for k in 0..TEMP_PROD_DCONV_DEPTH {
        let dilation: usize = 1 << k;
        let dc_padding = dilation * (TEMP_PROD_DCONV_KERNEL - 1) / 2;

        let cw = b.add_input(
            &format!("dc{k}_cw"),
            &[compressed, out_ch, TEMP_PROD_DCONV_KERNEL],
        );
        let cb = b.add_input(&format!("dc{k}_cb"), &[compressed]);
        let ng = b.add_input(&format!("dc{k}_ng"), &[compressed]);
        let nb = b.add_input(&format!("dc{k}_nb"), &[compressed]);
        let ew = b.add_input(&format!("dc{k}_ew"), &[doubled, compressed, 1]);
        let eb = b.add_input(&format!("dc{k}_eb"), &[doubled]);
        let eng = b.add_input(&format!("dc{k}_eng"), &[doubled]);
        let enb = b.add_input(&format!("dc{k}_enb"), &[doubled]);
        let ls_node = b.add_input(&format!("dc{k}_ls"), &[out_ch]);
        let eps1 = b.add_input(&format!("dc{k}_eps1"), &[1]);
        let eps2 = b.add_input(&format!("dc{k}_eps2"), &[1]);

        // Dilated Conv1d
        let c1 = b.add_conv1d_full(
            dconv_out,
            cw,
            Some(cb),
            1,
            dc_padding,
            dilation,
            1,
            &[compressed, conv_t_out],
        );
        let n1 = b.add_group_norm_g1(c1, eps1, Some(ng), Some(nb), compressed, conv_t_out);
        let g1 = b.add_gelu(n1, &[compressed, conv_t_out]);
        let c2 = b.add_conv1d(g1, ew, Some(eb), 1, 0, &[doubled, conv_t_out]);
        let n2 = b.add_group_norm_g1(c2, eps2, Some(eng), Some(enb), doubled, conv_t_out);
        let glu = b.add_glu(n2, 0, &[doubled, conv_t_out]).expect("even dim");
        let ls_out = b.add_layer_scale(glu, ls_node, &[out_ch, conv_t_out]);
        dconv_out = b.add_binary_add(dconv_out, ls_out, &[out_ch, conv_t_out]);
    }

    // Rewrite: Conv1d(k=1) -> GLU
    let rw_weight = b.add_input("rw_w", &[doubled, out_ch, 1]);
    let rw_bias = b.add_input("rw_b", &[doubled]);
    let rw_out = b.add_conv1d(
        dconv_out,
        rw_weight,
        Some(rw_bias),
        1,
        0,
        &[doubled, conv_t_out],
    );
    let output = b
        .add_glu(rw_out, 0, &[doubled, conv_t_out])
        .expect("even dim for GLU");

    let def = b.build(output).expect("valid temporal prod block0 graph");

    // Build bindings: data(Variable), conv_w, conv_b, then per-DConv, then rewrite
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable);
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_ch, in_ch, TEMP_PROD_CONV_KERNEL]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_ch]),
        0.0f32,
    )));

    for _k in 0..TEMP_PROD_DCONV_DEPTH {
        // cw, cb, ng, nb, ew, eb, eng, enb, ls, eps1, eps2
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[compressed, out_ch, TEMP_PROD_DCONV_KERNEL]),
            0.01f32,
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
            0.01f32,
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
            IxDyn(&[out_ch]),
            0.1f32,
        )));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    }

    // Rewrite weight + bias
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled, out_ch, 1]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        0.0f32,
    )));

    (def, conv_t_out, bindings)
}

/// Re-verify `demucs_temporal_encoder_prod_block0` with Conservative mode.
#[test]
fn test_sound_temporal_encoder_prod_block0() {
    let (def, conv_t_out, bindings) = build_temporal_prod_block0();
    let input = uniform_bounds(&[TEMP_PROD_IN_CH, TEMP_PROD_T_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "demucs_temporal_encoder_prod_block0",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Temporal encoder prod block0 (Conservative): bounds=[{lo}, {hi}], width={}, soundness={:?}",
        result.verification.output_width, result.verification.soundness_mode
    );

    let (lo_arr, _) = result.output_bounds.lower_upper();
    assert_eq!(
        lo_arr.shape(),
        &[TEMP_PROD_OUT_CH, conv_t_out],
        "output shape mismatch"
    );
}

// ===========================================================================
// 6. Temporal decoder block — Sound re-verification
// ===========================================================================

const TEMP_DEC_IN_CH: usize = 16;
const TEMP_DEC_OUT_CH: usize = 8;
const TEMP_DEC_T_IN: usize = 4;
const TEMP_DEC_CONV_KERNEL: usize = 8;
const TEMP_DEC_CONV_STRIDE: usize = 4;
const TEMP_DEC_CONV_PADDING: usize = 2;

fn build_temporal_decoder_block() -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    usize,
    Vec<TensorParamBinding>,
) {
    use super::common::conv_transpose_out_len;

    let in_ch = TEMP_DEC_IN_CH;
    let out_ch = TEMP_DEC_OUT_CH;
    let doubled = in_ch * 2;

    let mut b = TensorBlockBuilder::new("demucs_temp_dec_block_verify");

    let data = b.add_input("data", &[in_ch, TEMP_DEC_T_IN]);

    // GLU rewrite: Conv1d(k=1) doubles channels, then GLU halves
    let rw_weight = b.add_input("rw_w", &[doubled, in_ch, 1]);
    let rw_bias = b.add_input("rw_b", &[doubled]);

    let rw_out = b.add_conv1d(
        data,
        rw_weight,
        Some(rw_bias),
        1,
        0,
        &[doubled, TEMP_DEC_T_IN],
    );
    let glu_out = b
        .add_glu(rw_out, 0, &[doubled, TEMP_DEC_T_IN])
        .expect("even dim");

    // GELU activation
    let gelu_out = b.add_gelu(glu_out, &[in_ch, TEMP_DEC_T_IN]);

    // ConvTranspose1d upsample
    let tr_weight = b.add_input("tr_w", &[in_ch, out_ch, TEMP_DEC_CONV_KERNEL]);
    let tr_bias = b.add_input("tr_b", &[out_ch]);

    let t_out = conv_transpose_out_len(
        TEMP_DEC_T_IN,
        TEMP_DEC_CONV_STRIDE,
        TEMP_DEC_CONV_KERNEL,
        TEMP_DEC_CONV_PADDING,
    );

    let output = b.add_conv_transpose_1d(
        gelu_out,
        tr_weight,
        Some(tr_bias),
        TEMP_DEC_CONV_STRIDE,
        TEMP_DEC_CONV_PADDING,
        1,
        1,
        0,
        &[out_ch, t_out],
    );

    let def = b.build(output).expect("valid temporal decoder block graph");

    let mut bindings = vec![TensorParamBinding::Variable];
    // Rewrite weight + bias
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled, in_ch, 1]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        0.0f32,
    )));
    // ConvTranspose1d weight + bias
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[in_ch, out_ch, TEMP_DEC_CONV_KERNEL]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_ch]),
        0.0f32,
    )));

    (def, t_out, bindings)
}

/// Re-verify `demucs_temporal_decoder_block` with Conservative mode.
#[test]
fn test_sound_temporal_decoder_block() {
    let (def, t_out, bindings) = build_temporal_decoder_block();
    let input = uniform_bounds(&[TEMP_DEC_IN_CH, TEMP_DEC_T_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "demucs_temporal_decoder_block",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Temporal decoder block (Conservative): bounds=[{lo}, {hi}], width={}, soundness={:?}",
        result.verification.output_width, result.verification.soundness_mode
    );

    let (lo_arr, _) = result.output_bounds.lower_upper();
    assert_eq!(
        lo_arr.shape(),
        &[TEMP_DEC_OUT_CH, t_out],
        "output shape mismatch"
    );
}

// ===========================================================================
// 7. Cross-domain bottleneck — Sound re-verification
// ===========================================================================

// Cross-domain parameters (matching compose_demucs_cross_domain.rs)
const CD_ENC_CH: usize = 4;
const CD_MODEL_DIM: usize = 8;
const CD_NUM_HEADS: usize = 2;
const CD_FFN_HIDDEN: usize = CD_MODEL_DIM * 2;
const CD_T_SEQ: usize = 4;
const CD_F_SEQ: usize = CD_T_SEQ;
const CD_WEIGHT_MAG: f32 = 0.01;

fn push_weight(bindings: &mut Vec<TensorParamBinding>, shape: &[usize], val: f32) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(shape),
        val,
    )));
}

fn build_cross_domain_conservative(
) -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    let d = CD_MODEL_DIM;
    let ffn = CD_FFN_HIDDEN;

    let mut bb = TensorBlockBuilder::new("demucs_cross_domain_conservative");

    let temporal = bb.add_input("temporal_enc", &[CD_ENC_CH, CD_T_SEQ]);
    let spectral_kv = bb.add_input("spectral_kv", &[CD_F_SEQ, d]);

    let t_up_w = bb.add_input("t_up_w", &[d, CD_ENC_CH, 1]);
    let t_up_b = bb.add_input("t_up_b", &[d]);
    let t_down_w = bb.add_input("t_down_w", &[CD_ENC_CH, d, 1]);
    let t_down_b = bb.add_input("t_down_b", &[CD_ENC_CH]);
    let eps = bb.add_input("eps", &[1]);

    // Self-attention weights
    let sa_ln1_w = bb.add_input("sa_ln1_w", &[d]);
    let sa_ln1_b = bb.add_input("sa_ln1_b", &[d]);
    let sa_ln2_w = bb.add_input("sa_ln2_w", &[d]);
    let sa_ln2_b = bb.add_input("sa_ln2_b", &[d]);
    let sa_q_w = bb.add_input("sa_q_w", &[d, d]);
    let sa_k_w = bb.add_input("sa_k_w", &[d, d]);
    let sa_v_w = bb.add_input("sa_v_w", &[d, d]);
    let sa_out_w = bb.add_input("sa_out_w", &[d, d]);
    let sa_ffn1_w = bb.add_input("sa_ffn1_w", &[ffn, d]);
    let sa_ffn2_w = bb.add_input("sa_ffn2_w", &[d, ffn]);

    // Cross-attention weights (manual, no LN2 on constant KV)
    let ca_ln1_w = bb.add_input("ca_ln1_w", &[d]);
    let ca_ln1_b = bb.add_input("ca_ln1_b", &[d]);
    let ca_ln3_w = bb.add_input("ca_ln3_w", &[d]);
    let ca_ln3_b = bb.add_input("ca_ln3_b", &[d]);
    let ca_lnout_w = bb.add_input("ca_lnout_w", &[d]);
    let ca_lnout_b = bb.add_input("ca_lnout_b", &[d]);
    let ca_q_w = bb.add_input("ca_q_w", &[d, d]);
    let ca_k_w = bb.add_input("ca_k_w", &[d, d]);
    let ca_v_w = bb.add_input("ca_v_w", &[d, d]);
    let ca_out_w = bb.add_input("ca_out_w", &[d, d]);
    let ca_ffn1_w = bb.add_input("ca_ffn1_w", &[ffn, d]);
    let ca_ffn2_w = bb.add_input("ca_ffn2_w", &[d, ffn]);

    // Temporal channel bridge: [C, T] -> Conv1d(1x1) -> [D, T]
    let t_up = bb.add_conv1d(temporal, t_up_w, Some(t_up_b), 1, 0, &[d, CD_T_SEQ]);
    let t_td = bb.add_transpose(t_up, &[1, 0], &[CD_T_SEQ, d]);

    // Self-attention: [T, D] -> TransformerBlock -> [T, D]
    let sa_weights = nn_dsl::TransformerBlockWeights {
        ln1_weight: sa_ln1_w,
        ln1_bias: sa_ln1_b,
        ln2_weight: sa_ln2_w,
        ln2_bias: sa_ln2_b,
        q_weight: sa_q_w,
        k_weight: sa_k_w,
        v_weight: sa_v_w,
        out_weight: sa_out_w,
        ffn1_weight: sa_ffn1_w,
        ffn2_weight: sa_ffn2_w,
        eps,
    };
    let tc = nn_dsl::TransformerBlockConfig {
        num_heads: CD_NUM_HEADS,
        mask: nn_dsl::AttentionMask::Standard,
        ffn_hidden_dim: ffn,
    };
    let t_self = bb
        .add_transformer_block(t_td, &sa_weights, &tc)
        .expect("temporal self-attention");

    // Cross-attention (manual decomposition)
    let shape = [CD_T_SEQ, d];
    let ffn_shape = [CD_T_SEQ, ffn];

    let normed_q = bb.add_layer_norm(t_self, eps, 1, ca_ln1_w, ca_ln1_b, &shape);
    let attn = bb
        .add_multi_head_cross_attention(
            normed_q,
            spectral_kv,
            ca_q_w,
            ca_k_w,
            ca_v_w,
            ca_out_w,
            CD_NUM_HEADS,
            nn_dsl::AttentionMask::Standard,
            &shape,
        )
        .expect("cross-MHA temporal queries spectral");
    let residual1 = bb.add_binary_add(t_self, attn, &shape);

    let normed3 = bb.add_layer_norm(residual1, eps, 1, ca_ln3_w, ca_ln3_b, &shape);
    let ffn1 = bb.add_linear(normed3, ca_ffn1_w, None, &ffn_shape);
    let act = bb.add_gelu(ffn1, &ffn_shape);
    let ffn2 = bb.add_linear(act, ca_ffn2_w, None, &shape);
    let residual2 = bb.add_binary_add(residual1, ffn2, &shape);

    let t_cross = bb.add_layer_norm(residual2, eps, 1, ca_lnout_w, ca_lnout_b, &shape);

    let t_dt = bb.add_transpose(t_cross, &[1, 0], &[d, CD_T_SEQ]);
    let t_out = bb.add_conv1d(t_dt, t_down_w, Some(t_down_b), 1, 0, &[CD_ENC_CH, CD_T_SEQ]);

    let def = bb
        .build(t_out)
        .expect("valid cross-domain conservative graph");

    // Build bindings
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable); // temporal
    push_weight(&mut bindings, &[CD_F_SEQ, d], 0.1); // spectral_kv

    // Channel bridge
    push_weight(&mut bindings, &[d, CD_ENC_CH, 1], CD_WEIGHT_MAG);
    push_weight(&mut bindings, &[d], 0.0);
    push_weight(&mut bindings, &[CD_ENC_CH, d, 1], CD_WEIGHT_MAG);
    push_weight(&mut bindings, &[CD_ENC_CH], 0.0);
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps

    // Self-attention bindings
    push_weight(&mut bindings, &[d], 1.0);
    push_weight(&mut bindings, &[d], 0.0);
    push_weight(&mut bindings, &[d], 1.0);
    push_weight(&mut bindings, &[d], 0.0);
    push_weight(&mut bindings, &[d, d], CD_WEIGHT_MAG);
    push_weight(&mut bindings, &[d, d], CD_WEIGHT_MAG);
    push_weight(&mut bindings, &[d, d], CD_WEIGHT_MAG);
    push_weight(&mut bindings, &[d, d], CD_WEIGHT_MAG);
    push_weight(&mut bindings, &[ffn, d], CD_WEIGHT_MAG);
    push_weight(&mut bindings, &[d, ffn], CD_WEIGHT_MAG);

    // Cross-attention bindings (manual)
    push_weight(&mut bindings, &[d], 1.0);
    push_weight(&mut bindings, &[d], 0.0);
    push_weight(&mut bindings, &[d], 1.0);
    push_weight(&mut bindings, &[d], 0.0);
    push_weight(&mut bindings, &[d], 1.0);
    push_weight(&mut bindings, &[d], 0.0);
    push_weight(&mut bindings, &[d, d], CD_WEIGHT_MAG);
    push_weight(&mut bindings, &[d, d], CD_WEIGHT_MAG);
    push_weight(&mut bindings, &[d, d], CD_WEIGHT_MAG);
    push_weight(&mut bindings, &[d, d], CD_WEIGHT_MAG);
    push_weight(&mut bindings, &[ffn, d], CD_WEIGHT_MAG);
    push_weight(&mut bindings, &[d, ffn], CD_WEIGHT_MAG);

    (def, bindings)
}

/// Re-verify `demucs_cross_domain_bottleneck` with Conservative mode.
///
/// The cross-domain transformer bottleneck has LayerNorm layers that
/// produce heuristic classification in ForwardMode. Conservative IBP
/// avoids heuristic linearization entirely, producing Sound classification.
#[test]
fn test_sound_cross_domain_bottleneck() {
    let (def, bindings) = build_cross_domain_conservative();
    let input = uniform_bounds(&[CD_ENC_CH, CD_T_SEQ], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "demucs_cross_domain_bottleneck",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Cross-domain bottleneck (Conservative): bounds=[{lo}, {hi}], width={}, soundness={:?}",
        result.verification.output_width, result.verification.soundness_mode
    );

    let (lo_arr, _) = result.output_bounds.lower_upper();
    assert_eq!(
        lo_arr.shape(),
        &[CD_ENC_CH, CD_T_SEQ],
        "output must match temporal input shape"
    );
}
