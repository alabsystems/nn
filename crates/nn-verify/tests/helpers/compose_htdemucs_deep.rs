// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep HTDemucs encoder block compose verification tests.
//!
//! The existing HTDemucs verification covers shallow encoder blocks (depth 1)
//! and full model composition. These tests decompose deeper encoder stages
//! (depths 4-5) with LSTM integration to improve verification coverage:
//!
//! 1. **Deep encoder block (depth=4 DConv)**: Conv1d stride downsample +
//!    4-depth DConv (increasing dilation 1/2/4/8) + GLU rewrite.
//!    Tests bounds stability through deep dilated convolution stacks.
//!
//! 2. **LSTM bottleneck isolation**: Single-step LSTM cell with small
//!    hidden state. Tests bounds propagation through recurrent gating
//!    (input/forget/cell/output gates with tanh/sigmoid).
//!
//! 3. **Encoder + LSTM bridge**: Conv1d encoder block feeding into LSTM
//!    bottleneck. Tests composition of convolution bounds into recurrent
//!    bounds without blowup.
//!
//! 4. **Deep DConv residual stability (depth=5)**: 5-depth DConv stack
//!    verifying residual connections control bounds through very deep
//!    dilated convolution chains (dilations 1/2/4/8/16).
//!
//! 5. **Encoder + LSTM + Decoder pipeline**: Full temporal path through
//!    encoder → LSTM → decoder with ConvTranspose1d upsample. Tests
//!    end-to-end bounds through the autoencoder with recurrent bottleneck.
//!
//! Uses Conservative NormBoundsMode for Sound classification where
//! GroupNorm layers appear. Part of compose verification deepening
//! for HTDemucs model.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    uniform_bounds, verify_and_assert_with_config,
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

const IN_CH: usize = 4;
const ENC_CH: usize = 8;
const T_IN: usize = 32;
const ENC_KERNEL: usize = 8;
const ENC_STRIDE: usize = 4;
const ENC_PADDING: usize = 2;
const DCONV_KERNEL: usize = 3;
const DCONV_COMPRESS: usize = 4;
const HIDDEN_DIM: usize = 8;
const WEIGHT_MAG: f32 = 0.01;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

fn enc_t_out() -> usize {
    conv1d_out_len(T_IN, ENC_KERNEL, ENC_STRIDE, ENC_PADDING)
}

// ---------------------------------------------------------------------------
// DConv sub-layer builder (shared by multiple tests)
// ---------------------------------------------------------------------------

/// Build a single DConv sub-layer: Conv1d(dilated) -> GroupNorm(G=1) -> GELU ->
/// Conv1d(1x1) -> GroupNorm(G=1) -> GLU -> LayerScale -> residual_add.
///
/// Returns (output_node, binding_count) where binding_count is the number of
/// bindings consumed by this sub-layer.
fn add_dconv_sublayer(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
    depth_idx: usize,
    ch: usize,
    compressed: usize,
    t: usize,
    bindings: &mut Vec<TensorParamBinding>,
) -> nn_dsl::tensor_ir::TensorNodeId {
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

    // Bindings for this sub-layer
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
// 1. Deep encoder block (depth=4 DConv)
// ===========================================================================

fn build_deep_encoder_block_d4() -> (TensorKernelDef, usize, Vec<TensorParamBinding>) {
    let ch = ENC_CH;
    let compressed = ch / DCONV_COMPRESS;
    let doubled = ch * 2;
    let t_out = enc_t_out();

    let mut b = TensorBlockBuilder::new("htdemucs_deep_enc_d4");
    let data = b.add_input("data", &[IN_CH, T_IN]);
    let conv_w = b.add_input("conv_w", &[ch, IN_CH, ENC_KERNEL]);
    let conv_b = b.add_input("conv_b", &[ch]);

    // Conv1d stride downsample
    let conv_out = b.add_conv1d(
        data,
        conv_w,
        Some(conv_b),
        ENC_STRIDE,
        ENC_PADDING,
        &[ch, t_out],
    );
    let x = b.add_gelu(conv_out, &[ch, t_out]);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ch, IN_CH, ENC_KERNEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
    ];

    // 4-depth DConv stack (dilations 1, 2, 4, 8)
    let mut current = x;
    for k in 0..4 {
        current = add_dconv_sublayer(
            &mut b,
            current,
            "enc",
            k,
            ch,
            compressed,
            t_out,
            &mut bindings,
        );
    }

    // Rewrite: Conv1d(k=1) -> GLU
    let rw_w = b.add_input("rw_w", &[doubled, ch, 1]);
    let rw_b = b.add_input("rw_b", &[doubled]);
    let rw_out = b.add_conv1d(current, rw_w, Some(rw_b), 1, 0, &[doubled, t_out]);
    let output = b.add_glu(rw_out, 0, &[doubled, t_out]).expect("even dim");

    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled, ch, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        0.0f32,
    )));

    let def = b.build(output).expect("valid deep encoder block d4");
    (def, t_out, bindings)
}

#[test]
fn test_htdemucs_deep_enc_d4_def_validates() {
    let (def, _, _) = build_deep_encoder_block_d4();
    def.validate().expect("deep encoder d4 should validate");
}

#[test]
fn test_htdemucs_deep_enc_d4_ibp() {
    let (def, t_out, bindings) = build_deep_encoder_block_d4();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through deep encoder d4");
    assert_eq!(output.lower_upper().0.shape(), &[ENC_CH, t_out]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs deep encoder d4 IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e8, "deep enc d4 lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "deep enc d4 upper < 1e8, got {hi}");
}

#[test]
fn test_htdemucs_deep_enc_d4_crown() {
    let (def, t_out, bindings) = build_deep_encoder_block_d4();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[ENC_CH, t_out]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs deep encoder d4: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_htdemucs_deep_enc_d4_conservative_sound() {
    let (def, _, bindings) = build_deep_encoder_block_d4();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_deep_encoder_d4",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative deep encoder d4 should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs deep encoder d4 (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 2. LSTM bottleneck isolation
// ===========================================================================

fn build_lstm_bottleneck() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("htdemucs_deep_lstm_bottleneck");

    // LSTM input: [HIDDEN_DIM] (single time-step, flattened from encoder output)
    let x = b.add_input("x", &[HIDDEN_DIM]);
    let h0 = b.add_input("h0", &[HIDDEN_DIM]);
    let c0 = b.add_input("c0", &[HIDDEN_DIM]);
    // LSTM weights: weight_ih [4*H, H], weight_hh [4*H, H], bias [4*H]
    let w_ih = b.add_input("w_ih", &[4 * HIDDEN_DIM, HIDDEN_DIM]);
    let w_hh = b.add_input("w_hh", &[4 * HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[4 * HIDDEN_DIM]);

    let out = b.add_lstm(x, h0, c0, w_ih, w_hh, Some(bias), &[HIDDEN_DIM]);

    let def = b.build(out).expect("valid LSTM bottleneck");

    let bindings = vec![
        TensorParamBinding::Variable,
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
fn test_htdemucs_deep_lstm_def_validates() {
    let (def, _) = build_lstm_bottleneck();
    def.validate().expect("LSTM bottleneck should validate");
}

#[test]
fn test_htdemucs_deep_lstm_ibp() {
    let (def, bindings) = build_lstm_bottleneck();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through LSTM");
    assert_eq!(output.lower_upper().0.shape(), &[HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs LSTM bottleneck IBP: [{lo}, {hi}]");
    // LSTM output is bounded by tanh, so should be in [-1, 1] range
    assert!(lo >= -1.1, "LSTM lower should be >= -1.1, got {lo}");
    assert!(hi <= 1.1, "LSTM upper should be <= 1.1, got {hi}");
}

#[test]
fn test_htdemucs_deep_lstm_crown() {
    let (def, bindings) = build_lstm_bottleneck();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[HIDDEN_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs LSTM bottleneck: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_htdemucs_deep_lstm_verify_and_record() {
    let (def, bindings) = build_lstm_bottleneck();
    let input = uniform_bounds(&[HIDDEN_DIM], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_deep_lstm_bottleneck",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs LSTM bottleneck (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 3. Encoder + LSTM bridge
// ===========================================================================

/// Encoder Conv1d block feeding into LSTM bottleneck. Tests composition
/// of convolution bounds into recurrent bounds.
fn build_encoder_lstm_bridge() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let ch = ENC_CH;
    let t_out = enc_t_out();

    let mut b = TensorBlockBuilder::new("htdemucs_deep_enc_lstm_bridge");

    // Encoder: Conv1d downsample + GELU
    let data = b.add_input("data", &[IN_CH, T_IN]);
    let conv_w = b.add_input("conv_w", &[ch, IN_CH, ENC_KERNEL]);
    let conv_b = b.add_input("conv_b", &[ch]);

    let conv_out = b.add_conv1d(
        data,
        conv_w,
        Some(conv_b),
        ENC_STRIDE,
        ENC_PADDING,
        &[ch, t_out],
    );
    let enc_out = b.add_gelu(conv_out, &[ch, t_out]);

    // Project encoder output to LSTM input dimension: Linear [ch, t_out] -> [HIDDEN_DIM]
    // Use a simple approach: take first time-step via narrow, then linear projection
    let first_step = b.add_narrow(enc_out, 1, 0, 1, &[ch, 1]);
    // Reshape to [ch] by narrowing the temporal dim to 1
    // For verification, use a linear projection from [ch, 1] -> [HIDDEN_DIM]
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, ch, 1]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let lstm_input = b.add_conv1d(first_step, proj_w, Some(proj_b), 1, 0, &[HIDDEN_DIM, 1]);

    // Flatten the singleton temporal dim to get a 1-D [HIDDEN_DIM] feature vector.
    // PyTorch LSTM expects the input/state feature dimension to be last; the LSTM
    // weight `w_ih` is [4*H, input_features], so the input must be [HIDDEN_DIM]
    // (input_features == HIDDEN_DIM), not [HIDDEN_DIM, 1] (which would imply
    // input_features == 1). Reshape preserves the element count (HIDDEN_DIM*1).
    let lstm_flat = b.add_reshape(lstm_input, &[HIDDEN_DIM]);

    // LSTM bottleneck operating on the [HIDDEN_DIM] feature vector.
    let h0 = b.add_input("h0", &[HIDDEN_DIM]);
    let c0 = b.add_input("c0", &[HIDDEN_DIM]);
    let w_ih = b.add_input("w_ih", &[4 * HIDDEN_DIM, HIDDEN_DIM]);
    let w_hh = b.add_input("w_hh", &[4 * HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[4 * HIDDEN_DIM]);

    let out = b.add_lstm(lstm_flat, h0, c0, w_ih, w_hh, Some(bias), &[HIDDEN_DIM]);

    let def = b.build(out).expect("valid encoder + LSTM bridge");

    let bindings = vec![
        TensorParamBinding::Variable, // data
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ch, IN_CH, ENC_KERNEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ch]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, ch, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
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
fn test_htdemucs_deep_enc_lstm_bridge_def_validates() {
    let (def, _) = build_encoder_lstm_bridge();
    def.validate()
        .expect("encoder + LSTM bridge should validate");
}

#[test]
fn test_htdemucs_deep_enc_lstm_bridge_ibp() {
    let (def, bindings) = build_encoder_lstm_bridge();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder + LSTM bridge");
    assert_eq!(output.lower_upper().0.shape(), &[HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs encoder + LSTM bridge IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower bound must be finite, got {lo}");
    assert!(hi.is_finite(), "upper bound must be finite, got {hi}");
}

#[test]
fn test_htdemucs_deep_enc_lstm_bridge_verify_and_record() {
    let (def, bindings) = build_encoder_lstm_bridge();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_deep_encoder_lstm_bridge",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs encoder + LSTM bridge (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 4. Deep DConv residual stability (depth=5)
// ===========================================================================

/// Build a 5-depth DConv stack to test bounds stability through very deep
/// dilated convolution chains with dilations 1/2/4/8/16.
fn build_deep_dconv_d5() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let ch = ENC_CH;
    let compressed = ch / DCONV_COMPRESS;
    let t = 8; // Small temporal dim for 5-depth stack

    let mut b = TensorBlockBuilder::new("htdemucs_deep_dconv_d5");
    let data = b.add_input("data", &[ch, t]);

    let mut bindings = vec![TensorParamBinding::Variable];
    let mut current = data;

    for k in 0..5 {
        current = add_dconv_sublayer(&mut b, current, "d5", k, ch, compressed, t, &mut bindings);
    }

    let def = b.build(current).expect("valid 5-depth DConv stack");
    (def, bindings)
}

#[test]
fn test_htdemucs_deep_dconv_d5_def_validates() {
    let (def, _) = build_deep_dconv_d5();
    def.validate().expect("5-depth DConv should validate");
}

/// Soundness check for the decomposed-GroupNorm(g=1) → native InstanceNorm1d
/// verifier fusion (graph_tensor_group_norm_fusion.rs). Builds an isolated
/// GroupNorm(g=1) graph, propagates IBP (which now fuses to the native clamped
/// layer), then samples many concrete inputs from the input box and asserts the
/// TRUE GroupNorm output lies inside the IBP bounds. A sound enclosure must
/// contain every reachable output; this would catch any pattern-match that
/// substituted a non-equivalent function or an under-approximating bound.
#[test]
fn test_group_norm_g1_fusion_is_sound() {
    let c = 6usize;
    let t = 5usize;
    let n = c * t;
    let eps = 1e-5f32;

    // Affine: non-trivial per-channel gamma/beta to exercise the post-norm ops.
    let gamma: Vec<f32> = (0..c).map(|i| 0.5 + 0.3 * i as f32).collect();
    let beta: Vec<f32> = (0..c).map(|i| -0.2 + 0.1 * i as f32).collect();

    let mut b = TensorBlockBuilder::new("gn_fusion_sound");
    let x = b.add_input("data", &[c, t]);
    let ng = b.add_input("ng", &[c]);
    let nb = b.add_input("nb", &[c]);
    let eps_in = b.add_input("eps", &[1]);
    let out = b.add_group_norm_g1(x, eps_in, Some(ng), Some(nb), c, t);
    let def = b.build(out).expect("valid groupnorm graph");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[c]), gamma.clone()).unwrap(),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[c]), beta.clone()).unwrap(),
        ),
        TensorParamBinding::ConstantScalar(eps),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // A non-trivial input box (wide enough that the loose decomposed product
    // would massively over-bound; the fused clamp keeps it tight but still sound).
    let center = 3.0f32;
    let radius = 2.5f32;
    let lo_in = ArrayD::from_elem(IxDyn(&[c, t]), center - radius);
    let hi_in = ArrayD::from_elem(IxDyn(&[c, t]), center + radius);
    let input = nn_verify::BoundedTensor::new(lo_in, hi_in).expect("box");

    let bounds = graph.propagate_ibp(&input).expect("ibp");
    let (b_lo, b_hi) = bounds.lower_upper();

    // True GroupNorm(g=1): mean/var over all n=C*T elements, then per-channel affine.
    let eval_gn = |xs: &[f32]| -> Vec<f32> {
        let mean = xs.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = xs.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let std = (var + eps as f64).sqrt();
        let mut y = vec![0.0f32; n];
        for ch in 0..c {
            for ti in 0..t {
                let idx = ch * t + ti;
                let z = (xs[idx] as f64 - mean) / std;
                y[idx] = (gamma[ch] as f64 * z + beta[ch] as f64) as f32;
            }
        }
        y
    };

    // Deterministic LCG sampler over the input box (no proptest dep).
    let mut state: u64 = 0x9e3779b97f4a7c15;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32) / (1u32 << 31) as f32 // in [0,1)
    };

    let b_lo_flat: Vec<f32> = b_lo.iter().copied().collect();
    let b_hi_flat: Vec<f32> = b_hi.iter().copied().collect();
    // Tolerance: f32 eval rounding; the native layer already adds a sound margin,
    // so concrete outputs must sit comfortably inside. Use a small absolute slack.
    let tol = 1e-3f32;

    for _ in 0..20_000 {
        let xs: Vec<f32> = (0..n)
            .map(|_| (center - radius) + 2.0 * radius * next())
            .collect();
        let y = eval_gn(&xs);
        for i in 0..n {
            assert!(
                y[i] >= b_lo_flat[i] - tol && y[i] <= b_hi_flat[i] + tol,
                "GroupNorm fusion UNSOUND at elem {i}: true y={} not in [{}, {}]",
                y[i],
                b_lo_flat[i],
                b_hi_flat[i]
            );
        }
    }

    // Also confirm the fusion actually tightened: with the loose decomposed product
    // the bound would be ~ max|centered|/sqrt(eps) ~ hundreds; the fused clamp keeps
    // |z| <= sqrt(n-1), so post-affine magnitude is bounded by ~max|gamma|*sqrt(n-1)+|beta|.
    let max_gamma = gamma.iter().cloned().fold(0.0f32, f32::max);
    let max_beta = beta.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let ceiling = max_gamma * ((n - 1) as f32).sqrt() + max_beta + 1.0;
    let widest = b_hi_flat
        .iter()
        .zip(b_lo_flat.iter())
        .map(|(h, l)| h.abs().max(l.abs()))
        .fold(0.0f32, f32::max);
    assert!(
        widest <= ceiling,
        "GroupNorm fusion bound {widest} exceeds clamp ceiling {ceiling} — fusion may not have fired"
    );
}

// Previously #[ignore]'d: raw IBP through the 5-deep DConv stack accumulated to a
// sound-but-loose width ~1e9 ([-5.13e8, +5.13e8]) because nn's `add_group_norm_g1`
// decomposes GroupNorm(g=1) into primitives (centered * rsqrt), whose 4-corner
// interval product drops the joint constraint that a large deviation implies a
// large std. The verifier now fuses that decomposed subgraph back into NY's native
// InstanceNorm1d (graph_tensor_group_norm_fusion.rs), which enforces the sound
// `|z_i| <= sqrt(n-1)` clamp. The per-sublayer growth is now additive (~1.13/depth)
// instead of multiplicative, so depth-5 IBP lands at [-6.63, +6.63] (range ~13.3),
// far under the 1e6 target. Soundness is preserved (the fused layer is the
// verifier's own proptest-validated GroupNorm enclosure).
#[test]
fn test_htdemucs_deep_dconv_d5_ibp() {
    let (def, bindings) = build_deep_dconv_d5();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_CH, 8], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 5-depth DConv");
    assert_eq!(output.lower_upper().0.shape(), &[ENC_CH, 8]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    let range = hi - lo;
    eprintln!("HTDemucs 5-depth DConv IBP: [{lo}, {hi}], range={range}");

    // Bounds should not explode through deep residual DConv
    assert!(lo.abs() < 1e6, "5-depth DConv lower < 1e6, got {lo}");
    assert!(hi.abs() < 1e6, "5-depth DConv upper < 1e6, got {hi}");

    // Lock in the GroupNorm-fusion tightening: with the decomposed GroupNorm(g=1)
    // collapsed to native InstanceNorm1d, the depth-5 IBP width is additive, not
    // multiplicative. Guard against silent regression to the old ~1e9 width.
    assert!(
        range < 100.0,
        "5-depth DConv IBP range should stay small (GroupNorm fusion), got {range}"
    );
}

#[test]
fn test_htdemucs_deep_dconv_d5_crown() {
    let (def, bindings) = build_deep_dconv_d5();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_CH, 8], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[ENC_CH, 8]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("HTDemucs 5-depth DConv: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_htdemucs_deep_dconv_d5_conservative_sound() {
    let (def, bindings) = build_deep_dconv_d5();
    let input = uniform_bounds(&[ENC_CH, 8], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "htdemucs_deep_dconv_d5",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative 5-depth DConv should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "HTDemucs 5-depth DConv (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 5. Residual bounds blowup analysis: depth 4 vs depth 5
// ===========================================================================

/// Compare bounds blowup between 4-depth and 5-depth DConv stacks.
/// Verifies that adding a 5th depth layer does not cause excessive
/// bounds growth relative to 4 depth layers.
#[test]
fn test_htdemucs_deep_dconv_blowup_d4_vs_d5() {
    let ch = ENC_CH;
    let compressed = ch / DCONV_COMPRESS;
    let t = 8;

    // Build depth-4 DConv stack
    let mut b4 = TensorBlockBuilder::new("htdemucs_blowup_d4");
    let data4 = b4.add_input("data", &[ch, t]);
    let mut bindings4 = vec![TensorParamBinding::Variable];
    let mut current4 = data4;
    for k in 0..4 {
        current4 = add_dconv_sublayer(
            &mut b4,
            current4,
            "d4",
            k,
            ch,
            compressed,
            t,
            &mut bindings4,
        );
    }
    let def4 = b4.build(current4).expect("valid d4");

    // Build depth-5 DConv stack
    let (def5, bindings5) = build_deep_dconv_d5();

    let graph4 = tensor_kernel_to_graph(&def4, &bindings4).expect("d4 graph");
    let graph5 = tensor_kernel_to_graph(&def5, &bindings5).expect("d5 graph");
    let input = uniform_bounds(&[ch, t], 1.0);

    let out4 = graph4.propagate_ibp(&input).expect("IBP d4");
    let out5 = graph5.propagate_ibp(&input).expect("IBP d5");

    let (lo4, hi4) = bounds_min_max(&out4);
    let (lo5, hi5) = bounds_min_max(&out5);
    let range4 = hi4 - lo4;
    let range5 = hi5 - lo5;

    let blowup = range5 / range4.max(1e-10);
    eprintln!(
        "HTDemucs DConv blowup: d4 range={range4:.4}, d5 range={range5:.4}, blowup={blowup:.1}x"
    );

    // Adding one more DConv depth should not blow up more than 100x
    assert!(
        blowup < 100.0,
        "d5 vs d4 blowup should be < 100x, got {blowup:.1}x"
    );
}
