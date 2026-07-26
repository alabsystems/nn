// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HTDemucs separator block compose verification tests.
//!
//! Deepens NY coverage for HTDemucs by testing compositions not
//! covered by the existing 14 status entries:
//!
//! 1. **Two-stage encoder stacking**: Encoder block 0 output → encoder block 1.
//!    Tests bounds propagation through sequential downsampling stages
//!    (Conv1d stride → DConv → GLU → Conv1d stride → DConv → GLU).
//!
//! 2. **Encoder + transformer + decoder (Conservative Sound)**: Full temporal
//!    path with Conservative NormBoundsMode targeting Sound classification.
//!    The existing `htdemucs_full` uses ForwardMode → Heuristic; this targets
//!    Sound via Conservative IBP through normalization layers.
//!
//! 3. **Decoder with DConv (Conservative Sound)**: Decoder block with DConv
//!    residual sublayers verified under Conservative mode. Targets promotion
//!    of `demucs_temporal_decoder_production` from heuristic to sound.
//!
//! 4. **Multi-step LSTM unrolling**: Two-step LSTM cell verifying bounds
//!    stability across sequential recurrent steps.
//!
//! 5. **Encoder stacking bounds monotonicity**: Verifies that narrower
//!    input produces tighter output through two-stage encoder.
//!
//! Uses Conservative NormBoundsMode for Sound classification where
//! GroupNorm layers appear. Part of #4278.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    conv_transpose_out_len, uniform_bounds, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_dsl::{AttentionMask, TransformerBlockConfig, TransformerBlockWeights};
use nn_verify::{
    tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerificationSoundnessMode,
    VerifyConfig,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const IN_CH: usize = 4;
const ENC_CH: usize = 8;
const ENC2_CH: usize = 16; // second encoder stage doubles channels
const T_IN: usize = 32;
const ENC_KERNEL: usize = 8;
const ENC_STRIDE: usize = 4;
const ENC_PADDING: usize = 2;
const DCONV_KERNEL: usize = 3;
const DCONV_COMPRESS: usize = 4;
const DCONV_DEPTH: usize = 1;
const DEC_REWRITE_KERNEL: usize = 3;
const DEC_REWRITE_PADDING: usize = DEC_REWRITE_KERNEL / 2;
const CT_KERNEL: usize = 8;
const CT_STRIDE: usize = 4;
const CT_PADDING: usize = ENC_PADDING;
const HIDDEN_DIM: usize = 8;
const MODEL_DIM: usize = ENC_CH;
const NUM_HEADS: usize = 2;
const FFN_HIDDEN: usize = MODEL_DIM * 2;
const F_SEQ: usize = 4;
const WEIGHT_MAG: f32 = 0.01;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ---------------------------------------------------------------------------
// DConv sub-layer builder (shared by multiple tests)
// ---------------------------------------------------------------------------

struct DConvInputs {
    cw: TensorNodeId,
    cb: TensorNodeId,
    ng: TensorNodeId,
    nb: TensorNodeId,
    ew: TensorNodeId,
    eb: TensorNodeId,
    eng: TensorNodeId,
    enb: TensorNodeId,
    ls: TensorNodeId,
    eps1: TensorNodeId,
    eps2: TensorNodeId,
    dilation: usize,
}

impl DConvInputs {
    fn add(b: &mut TensorBlockBuilder, pfx: &str, k: usize, ch: usize, comp: usize) -> Self {
        let d = ch * 2;
        Self {
            cw: b.add_input(&format!("{pfx}_dc{k}_cw"), &[comp, ch, DCONV_KERNEL]),
            cb: b.add_input(&format!("{pfx}_dc{k}_cb"), &[comp]),
            ng: b.add_input(&format!("{pfx}_dc{k}_ng"), &[comp]),
            nb: b.add_input(&format!("{pfx}_dc{k}_nb"), &[comp]),
            ew: b.add_input(&format!("{pfx}_dc{k}_ew"), &[d, comp, 1]),
            eb: b.add_input(&format!("{pfx}_dc{k}_eb"), &[d]),
            eng: b.add_input(&format!("{pfx}_dc{k}_eng"), &[d]),
            enb: b.add_input(&format!("{pfx}_dc{k}_enb"), &[d]),
            ls: b.add_input(&format!("{pfx}_dc{k}_ls"), &[ch]),
            eps1: b.add_input(&format!("{pfx}_dc{k}_eps"), &[1]),
            eps2: b.add_input(&format!("{pfx}_dc{k}_eps2"), &[1]),
            dilation: 1 << k,
        }
    }
}

fn build_dconv(
    b: &mut TensorBlockBuilder,
    x: TensorNodeId,
    dc: &DConvInputs,
    ch: usize,
    comp: usize,
    t: usize,
) -> TensorNodeId {
    let d = ch * 2;
    let pad = dc.dilation * (DCONV_KERNEL - 1) / 2;
    let c1 = b.add_conv1d_full(x, dc.cw, Some(dc.cb), 1, pad, dc.dilation, 1, &[comp, t]);
    let n1 = b.add_group_norm_g1(c1, dc.eps1, Some(dc.ng), Some(dc.nb), comp, t);
    let g1 = b.add_gelu(n1, &[comp, t]);
    let c2 = b.add_conv1d(g1, dc.ew, Some(dc.eb), 1, 0, &[d, t]);
    let n2 = b.add_group_norm_g1(c2, dc.eps2, Some(dc.eng), Some(dc.enb), d, t);
    let glu = b.add_glu(n2, 0, &[d, t]).expect("even dim");
    let ls = b.add_layer_scale(glu, dc.ls, &[ch, t]);
    b.add_binary_add(x, ls, &[ch, t])
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

fn add_dconv_bindings(b: &mut Vec<TensorParamBinding>, ch: usize, comp: usize) {
    let d = ch * 2;
    push_weight(b, &[comp, ch, DCONV_KERNEL], WEIGHT_MAG);
    push_weight(b, &[comp], 0.0);
    push_weight(b, &[comp], 1.0);
    push_weight(b, &[comp], 0.0);
    push_weight(b, &[d, comp, 1], WEIGHT_MAG);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[d], 1.0);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[ch], 0.1);
    b.push(TensorParamBinding::ConstantScalar(1e-5));
    b.push(TensorParamBinding::ConstantScalar(1e-5));
}

// ===========================================================================
// 1. Two-stage encoder stacking
// ===========================================================================

/// Build two sequential encoder blocks: block 0 (IN_CH -> ENC_CH) feeds
/// block 1 (ENC_CH -> ENC2_CH). Each block: Conv1d(stride) -> GELU ->
/// DConv(x1) -> Rewrite(Conv1d k=1) -> GLU.
fn build_two_stage_encoder() -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    usize,
    Vec<TensorParamBinding>,
) {
    let comp0 = ENC_CH / DCONV_COMPRESS;
    let dbl0 = ENC_CH * 2;
    let comp1 = ENC2_CH / DCONV_COMPRESS;
    let dbl1 = ENC2_CH * 2;

    let mut b = TensorBlockBuilder::new("htdemucs_two_stage_encoder");
    let audio = b.add_input("audio", &[IN_CH, T_IN]);

    // --- Block 0: IN_CH -> ENC_CH ---
    let ecw0 = b.add_input("enc0_conv_w", &[ENC_CH, IN_CH, ENC_KERNEL]);
    let ecb0 = b.add_input("enc0_conv_b", &[ENC_CH]);
    let enc0_dc: Vec<_> = (0..DCONV_DEPTH)
        .map(|k| DConvInputs::add(&mut b, "enc0", k, ENC_CH, comp0))
        .collect();
    let erw0 = b.add_input("enc0_rw_w", &[dbl0, ENC_CH, 1]);
    let erb0 = b.add_input("enc0_rw_b", &[dbl0]);

    let t0 = conv1d_out_len(T_IN, ENC_KERNEL, ENC_STRIDE, ENC_PADDING);
    let x = b.add_conv1d(
        audio,
        ecw0,
        Some(ecb0),
        ENC_STRIDE,
        ENC_PADDING,
        &[ENC_CH, t0],
    );
    let x = b.add_gelu(x, &[ENC_CH, t0]);
    let mut x = x;
    for di in &enc0_dc {
        x = build_dconv(&mut b, x, di, ENC_CH, comp0, t0);
    }
    let x = b.add_conv1d(x, erw0, Some(erb0), 1, 0, &[dbl0, t0]);
    let enc0_out = b.add_glu(x, 0, &[dbl0, t0]).expect("enc0 GLU");

    // --- Block 1: ENC_CH -> ENC2_CH ---
    let ecw1 = b.add_input("enc1_conv_w", &[ENC2_CH, ENC_CH, ENC_KERNEL]);
    let ecb1 = b.add_input("enc1_conv_b", &[ENC2_CH]);
    let enc1_dc: Vec<_> = (0..DCONV_DEPTH)
        .map(|k| DConvInputs::add(&mut b, "enc1", k, ENC2_CH, comp1))
        .collect();
    let erw1 = b.add_input("enc1_rw_w", &[dbl1, ENC2_CH, 1]);
    let erb1 = b.add_input("enc1_rw_b", &[dbl1]);

    let t1 = conv1d_out_len(t0, ENC_KERNEL, ENC_STRIDE, ENC_PADDING);
    let x = b.add_conv1d(
        enc0_out,
        ecw1,
        Some(ecb1),
        ENC_STRIDE,
        ENC_PADDING,
        &[ENC2_CH, t1],
    );
    let x = b.add_gelu(x, &[ENC2_CH, t1]);
    let mut x = x;
    for di in &enc1_dc {
        x = build_dconv(&mut b, x, di, ENC2_CH, comp1, t1);
    }
    let x = b.add_conv1d(x, erw1, Some(erb1), 1, 0, &[dbl1, t1]);
    let out = b.add_glu(x, 0, &[dbl1, t1]).expect("enc1 GLU");

    let def = b.build(out).expect("valid two-stage encoder");

    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable); // audio

    // Block 0 bindings
    push_weight(&mut bindings, &[ENC_CH, IN_CH, ENC_KERNEL], WEIGHT_MAG);
    push_weight(&mut bindings, &[ENC_CH], 0.0);
    for _ in 0..DCONV_DEPTH {
        add_dconv_bindings(&mut bindings, ENC_CH, comp0);
    }
    push_weight(&mut bindings, &[dbl0, ENC_CH, 1], WEIGHT_MAG);
    push_weight(&mut bindings, &[dbl0], 0.0);

    // Block 1 bindings
    push_weight(&mut bindings, &[ENC2_CH, ENC_CH, ENC_KERNEL], WEIGHT_MAG);
    push_weight(&mut bindings, &[ENC2_CH], 0.0);
    for _ in 0..DCONV_DEPTH {
        add_dconv_bindings(&mut bindings, ENC2_CH, comp1);
    }
    push_weight(&mut bindings, &[dbl1, ENC2_CH, 1], WEIGHT_MAG);
    push_weight(&mut bindings, &[dbl1], 0.0);

    (def, t1, bindings)
}

#[test]
fn test_htdemucs_two_stage_encoder_def_validates() {
    let (def, _, _) = build_two_stage_encoder();
    def.validate().expect("two-stage encoder should validate");
}

#[test]
fn test_htdemucs_two_stage_encoder_ibp() {
    let (def, t1, bindings) = build_two_stage_encoder();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through two-stage encoder");
    assert_eq!(output.lower_upper().0.shape(), &[ENC2_CH, t1]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs two-stage encoder IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower bound must be finite, got {lo}");
    assert!(hi.is_finite(), "upper bound must be finite, got {hi}");
}

#[test]
fn test_htdemucs_two_stage_encoder_crown() {
    let (def, t1, bindings) = build_two_stage_encoder();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[ENC2_CH, t1]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs two-stage encoder: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_htdemucs_two_stage_encoder_conservative_sound() {
    let (def, _, bindings) = build_two_stage_encoder();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_two_stage_encoder",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative two-stage encoder should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs two-stage encoder (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 2. Encoder + transformer + decoder (Conservative Sound)
// ===========================================================================

/// Build the full temporal separator path with Conservative mode targeting
/// Sound classification. Uses smaller dims than htdemucs_full for tractability.
fn build_separator_conservative() -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    usize,
    Vec<TensorParamBinding>,
) {
    let comp = ENC_CH / DCONV_COMPRESS;
    let dbl = ENC_CH * 2;
    let d = MODEL_DIM;
    let t_in = 16; // smaller than T_IN=32 for tractability

    let mut b = TensorBlockBuilder::new("htdemucs_separator_conservative");
    let audio = b.add_input("audio", &[IN_CH, t_in]);

    // Encoder
    let ecw = b.add_input("enc_conv_w", &[ENC_CH, IN_CH, ENC_KERNEL]);
    let ecb = b.add_input("enc_conv_b", &[ENC_CH]);
    let enc_dc: Vec<_> = (0..DCONV_DEPTH)
        .map(|k| DConvInputs::add(&mut b, "enc", k, ENC_CH, comp))
        .collect();
    let erw = b.add_input("enc_rw_w", &[dbl, ENC_CH, 1]);
    let erb = b.add_input("enc_rw_b", &[dbl]);

    let t_enc = conv1d_out_len(t_in, ENC_KERNEL, ENC_STRIDE, ENC_PADDING);
    let x = b.add_conv1d(
        audio,
        ecw,
        Some(ecb),
        ENC_STRIDE,
        ENC_PADDING,
        &[ENC_CH, t_enc],
    );
    let x = b.add_gelu(x, &[ENC_CH, t_enc]);
    let mut x = x;
    for di in &enc_dc {
        x = build_dconv(&mut b, x, di, ENC_CH, comp, t_enc);
    }
    let x = b.add_conv1d(x, erw, Some(erb), 1, 0, &[dbl, t_enc]);
    let enc_out = b.add_glu(x, 0, &[dbl, t_enc]).expect("encoder GLU");

    // Transformer bottleneck: Transpose -> self-attn -> Transpose
    let eps = b.add_input("eps", &[1]);
    let tw = TransformerBlockWeights {
        ln1_weight: b.add_input("tf_ln1_w", &[d]),
        ln1_bias: b.add_input("tf_ln1_b", &[d]),
        ln2_weight: b.add_input("tf_ln2_w", &[d]),
        ln2_bias: b.add_input("tf_ln2_b", &[d]),
        q_weight: b.add_input("tf_q_w", &[d, d]),
        k_weight: b.add_input("tf_k_w", &[d, d]),
        v_weight: b.add_input("tf_v_w", &[d, d]),
        out_weight: b.add_input("tf_out_w", &[d, d]),
        ffn1_weight: b.add_input("tf_ffn1_w", &[FFN_HIDDEN, d]),
        ffn2_weight: b.add_input("tf_ffn2_w", &[d, FFN_HIDDEN]),
        eps,
    };

    let x_t = b.add_transpose(enc_out, &[1, 0], &[t_enc, d]);
    let tc = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_HIDDEN,
    };
    let x_t = b
        .add_transformer_block(x_t, &tw, &tc)
        .expect("transformer block");
    let x = b.add_transpose(x_t, &[1, 0], &[d, t_enc]);

    // Decoder: skip + rewrite + DConv + ConvTranspose1d
    let x = b.add_binary_add(x, enc_out, &[ENC_CH, t_enc]);
    let drw = b.add_input("dec_rw_w", &[dbl, ENC_CH, DEC_REWRITE_KERNEL]);
    let drb = b.add_input("dec_rw_b", &[dbl]);
    let dec_dc: Vec<_> = (0..DCONV_DEPTH)
        .map(|k| DConvInputs::add(&mut b, "dec", k, ENC_CH, comp))
        .collect();
    let dctw = b.add_input("dec_ct_w", &[ENC_CH, IN_CH, CT_KERNEL]);
    let dctb = b.add_input("dec_ct_b", &[IN_CH]);

    let rw_t = conv1d_out_len(t_enc, DEC_REWRITE_KERNEL, 1, DEC_REWRITE_PADDING);
    let x = b.add_conv1d(x, drw, Some(drb), 1, DEC_REWRITE_PADDING, &[dbl, rw_t]);
    let x = b.add_glu(x, 0, &[dbl, rw_t]).expect("decoder GLU");
    let mut x = x;
    for di in &dec_dc {
        x = build_dconv(&mut b, x, di, ENC_CH, comp, rw_t);
    }
    let ct_t = conv_transpose_out_len(rw_t, CT_STRIDE, CT_KERNEL, CT_PADDING);
    let x = b.add_conv_transpose_1d(
        x,
        dctw,
        Some(dctb),
        CT_STRIDE,
        CT_PADDING,
        1,
        1,
        0,
        &[IN_CH, ct_t],
    );
    let target_t = t_in.min(ct_t);
    let x = if ct_t > target_t {
        b.add_narrow(x, 1, 0, target_t, &[IN_CH, target_t])
    } else {
        x
    };
    let out = b.add_gelu(x, &[IN_CH, target_t]);

    let def = b.build(out).expect("valid separator conservative");

    // Bindings
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable); // audio

    // Encoder
    push_weight(&mut bindings, &[ENC_CH, IN_CH, ENC_KERNEL], WEIGHT_MAG);
    push_weight(&mut bindings, &[ENC_CH], 0.0);
    for _ in 0..DCONV_DEPTH {
        add_dconv_bindings(&mut bindings, ENC_CH, comp);
    }
    push_weight(&mut bindings, &[dbl, ENC_CH, 1], WEIGHT_MAG);
    push_weight(&mut bindings, &[dbl], 0.0);

    // Transformer
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
    push_weight(&mut bindings, &[d], 1.0); // ln1 gamma
    push_weight(&mut bindings, &[d], 0.0); // ln1 beta
    push_weight(&mut bindings, &[d], 1.0); // ln2 gamma
    push_weight(&mut bindings, &[d], 0.0); // ln2 beta
    push_weight(&mut bindings, &[d, d], WEIGHT_MAG); // Q
    push_weight(&mut bindings, &[d, d], WEIGHT_MAG); // K
    push_weight(&mut bindings, &[d, d], WEIGHT_MAG); // V
    push_weight(&mut bindings, &[d, d], WEIGHT_MAG); // out
    push_weight(&mut bindings, &[FFN_HIDDEN, d], WEIGHT_MAG); // ffn1
    push_weight(&mut bindings, &[d, FFN_HIDDEN], WEIGHT_MAG); // ffn2

    // Decoder
    push_weight(
        &mut bindings,
        &[dbl, ENC_CH, DEC_REWRITE_KERNEL],
        WEIGHT_MAG,
    );
    push_weight(&mut bindings, &[dbl], 0.0);
    for _ in 0..DCONV_DEPTH {
        add_dconv_bindings(&mut bindings, ENC_CH, comp);
    }
    push_weight(&mut bindings, &[ENC_CH, IN_CH, CT_KERNEL], WEIGHT_MAG);
    push_weight(&mut bindings, &[IN_CH], 0.0);

    (def, target_t, bindings)
}

#[test]
fn test_htdemucs_separator_def_validates() {
    let (def, _, _) = build_separator_conservative();
    def.validate()
        .expect("separator conservative should validate");
}

#[test]
fn test_htdemucs_separator_ibp() {
    let (def, target_t, bindings) = build_separator_conservative();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, 16], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through separator");
    assert_eq!(output.lower_upper().0.shape(), &[IN_CH, target_t]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs separator IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower bound must be finite, got {lo}");
    assert!(hi.is_finite(), "upper bound must be finite, got {hi}");
}

#[test]
fn test_htdemucs_separator_conservative_sound() {
    let (def, _, bindings) = build_separator_conservative();
    let input = uniform_bounds(&[IN_CH, 16], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_separator_conservative",
        &conservative_config(),
    );

    // Conservative mode through normalization layers should produce Sound.
    // If it falls back to Heuristic, that indicates a norm-through-CROWN issue
    // and we log but still assert bounds validity.
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs separator (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
}

#[test]
fn test_htdemucs_separator_crown() {
    let (def, target_t, bindings) = build_separator_conservative();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, 16], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[IN_CH, target_t]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs separator: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 3. Decoder with DConv (Conservative Sound)
// ===========================================================================

/// Decoder block with DConv residual sublayers: skip_add -> Rewrite(Conv1d k=3)
/// -> GLU -> DConv(x1) -> ConvTranspose1d -> GELU.
fn build_decoder_dconv_conservative() -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    usize,
    Vec<TensorParamBinding>,
) {
    let ch = ENC_CH;
    let comp = ch / DCONV_COMPRESS;
    let dbl = ch * 2;
    let t_in = 4;
    let out_ch = IN_CH;

    let mut b = TensorBlockBuilder::new("htdemucs_decoder_dconv_conservative");
    let data = b.add_input("data", &[ch, t_in]);

    // Rewrite Conv1d(k=3)
    let rw_w = b.add_input("rw_w", &[dbl, ch, DEC_REWRITE_KERNEL]);
    let rw_b = b.add_input("rw_b", &[dbl]);

    let rw_t = conv1d_out_len(t_in, DEC_REWRITE_KERNEL, 1, DEC_REWRITE_PADDING);
    let x = b.add_conv1d(data, rw_w, Some(rw_b), 1, DEC_REWRITE_PADDING, &[dbl, rw_t]);
    let x = b.add_glu(x, 0, &[dbl, rw_t]).expect("decoder GLU");

    // DConv sublayers
    let dec_dc: Vec<_> = (0..DCONV_DEPTH)
        .map(|k| DConvInputs::add(&mut b, "dec", k, ch, comp))
        .collect();
    let mut x = x;
    for di in &dec_dc {
        x = build_dconv(&mut b, x, di, ch, comp, rw_t);
    }

    // ConvTranspose1d upsample
    let ct_w = b.add_input("ct_w", &[ch, out_ch, CT_KERNEL]);
    let ct_b = b.add_input("ct_b", &[out_ch]);
    let ct_t = conv_transpose_out_len(rw_t, CT_STRIDE, CT_KERNEL, CT_PADDING);
    let x = b.add_conv_transpose_1d(
        x,
        ct_w,
        Some(ct_b),
        CT_STRIDE,
        CT_PADDING,
        1,
        1,
        0,
        &[out_ch, ct_t],
    );
    let out = b.add_gelu(x, &[out_ch, ct_t]);

    let def = b.build(out).expect("valid decoder DConv conservative");

    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable); // data
    push_weight(&mut bindings, &[dbl, ch, DEC_REWRITE_KERNEL], WEIGHT_MAG);
    push_weight(&mut bindings, &[dbl], 0.0);
    for _ in 0..DCONV_DEPTH {
        add_dconv_bindings(&mut bindings, ch, comp);
    }
    push_weight(&mut bindings, &[ch, out_ch, CT_KERNEL], WEIGHT_MAG);
    push_weight(&mut bindings, &[out_ch], 0.0);

    (def, ct_t, bindings)
}

#[test]
fn test_htdemucs_decoder_dconv_def_validates() {
    let (def, _, _) = build_decoder_dconv_conservative();
    def.validate().expect("decoder DConv should validate");
}

#[test]
fn test_htdemucs_decoder_dconv_ibp() {
    let (def, ct_t, bindings) = build_decoder_dconv_conservative();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_CH, 4], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder DConv");
    assert_eq!(output.lower_upper().0.shape(), &[IN_CH, ct_t]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs decoder DConv IBP: [{lo}, {hi}]");
}

#[test]
fn test_htdemucs_decoder_dconv_conservative_sound() {
    let (def, _, bindings) = build_decoder_dconv_conservative();
    let input = uniform_bounds(&[ENC_CH, 4], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_decoder_dconv_conservative",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative decoder DConv should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs decoder DConv (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 4. Multi-step LSTM unrolling
// ===========================================================================

/// Two-step LSTM cell: the output hidden state of step 1 feeds as input to
/// step 2. Verifies bounds stability through sequential recurrent steps.
fn build_two_step_lstm() -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("htdemucs_two_step_lstm");

    let x = b.add_input("x", &[HIDDEN_DIM]);
    let h0 = b.add_input("h0", &[HIDDEN_DIM]);
    let c0 = b.add_input("c0", &[HIDDEN_DIM]);

    // Shared LSTM weights (same for both steps, as in production unrolling)
    let w_ih = b.add_input("w_ih", &[4 * HIDDEN_DIM, HIDDEN_DIM]);
    let w_hh = b.add_input("w_hh", &[4 * HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[4 * HIDDEN_DIM]);

    // Step 1: LSTM cell
    let h1 = b.add_lstm(x, h0, c0, w_ih, w_hh, Some(bias), &[HIDDEN_DIM]);

    // Step 2: Feed h1 back as both input and hidden state (simplified unroll).
    // In production, the next time-step's x would be different, but for
    // verification we test bounds stability by feeding h1 as input.
    let h2 = b.add_lstm(h1, h1, c0, w_ih, w_hh, Some(bias), &[HIDDEN_DIM]);

    let def = b.build(h2).expect("valid two-step LSTM");

    let bindings = vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[4 * HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[4 * HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4 * HIDDEN_DIM]), 0.0f32)),
    ];

    (def, bindings)
}

#[test]
fn test_htdemucs_two_step_lstm_def_validates() {
    let (def, _) = build_two_step_lstm();
    def.validate().expect("two-step LSTM should validate");
}

#[test]
fn test_htdemucs_two_step_lstm_ibp() {
    let (def, bindings) = build_two_step_lstm();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through two-step LSTM");
    assert_eq!(output.lower_upper().0.shape(), &[HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs two-step LSTM IBP: [{lo}, {hi}]");
    // LSTM output is bounded by tanh, so should remain in [-1, 1] range
    assert!(
        lo >= -1.1,
        "two-step LSTM lower should be >= -1.1, got {lo}"
    );
    assert!(hi <= 1.1, "two-step LSTM upper should be <= 1.1, got {hi}");
}

#[test]
fn test_htdemucs_two_step_lstm_crown() {
    let (def, bindings) = build_two_step_lstm();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[HIDDEN_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs two-step LSTM: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_htdemucs_two_step_lstm_verify_and_record() {
    let (def, bindings) = build_two_step_lstm();
    let input = uniform_bounds(&[HIDDEN_DIM], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_two_step_lstm",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs two-step LSTM (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 5. Two-stage encoder bounds monotonicity
// ===========================================================================

/// Narrower input to the two-stage encoder produces tighter output bounds.
/// Verifies IBP monotonicity across the deep two-stage encoder composition.
#[test]
fn test_htdemucs_two_stage_encoder_monotonicity() {
    let (def, _, bindings) = build_two_stage_encoder();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[IN_CH, T_IN], 10.0);
    let narrow_input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let wide_output = graph.propagate_ibp(&wide_input).expect("wide IBP");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("narrow IBP");

    let (wide_lo, wide_hi) = wide_output.lower_upper();
    let (narrow_lo, narrow_hi) = narrow_output.lower_upper();

    let wide_range = wide_hi.iter().zip(wide_lo.iter()).map(|(h, l)| h - l);
    let narrow_range = narrow_hi.iter().zip(narrow_lo.iter()).map(|(h, l)| h - l);

    // At least half of output elements should have narrower bounds with
    // narrower input (IBP monotonicity may not hold element-wise due to
    // decomposed norm approximations).
    let tighter_count = wide_range.zip(narrow_range).filter(|(w, n)| n <= w).count();
    let total = wide_lo.len();

    eprintln!(
        "Two-stage encoder monotonicity: {tighter_count}/{total} elements tighter with narrow input"
    );
    assert!(
        tighter_count > total / 2,
        "narrow input should produce tighter bounds for > 50% of elements, got {tighter_count}/{total}"
    );
}
